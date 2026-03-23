from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict, is_dataclass
import yaml
import numbers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

from environment.vec import Vector
from environment.immutables import MovementDim, BehaviorState
from environment.behaviors import (
    CorrelatedRandomWalk,
    CRW_CFG,
    ExploreExploit,
    EE_CFG,
    PointOfInterest,
    POI_CFG,
    LearningPointOfInterest,
    LPOI_CFG,
)


# ============================================================
# IO
# ============================================================

def load_segments_df(manifest_path: str | Path):
    manifest_path = Path(manifest_path)
    manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
    base_dir = manifest_path.parent

    frames = []
    for _, row in manifest.iterrows():
        seg_path = base_dir / row["path"]
        with np.load(seg_path, allow_pickle=False) as z:
            t = z["t"].astype(float)
            x = z["x"].astype(float)
            y = z["y"].astype(float)

        frames.append(pd.DataFrame({
            "segment": int(row["segment"]),
            "t": t,
            "x": x,
            "y": y,
        }))

    df = pd.concat(frames, ignore_index=True)
    return manifest, df

def _to_builtin(obj):
    """Recursively convert config-like objects to YAML-safe builtin types."""
    if is_dataclass(obj):
        return _to_builtin(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        return float(obj) if isinstance(obj, np.floating) else obj

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            str(k): _to_builtin(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return obj


def save_configs_yaml(path: str | Path, **configs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: _to_builtin(cfg) for name, cfg in configs.items()}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

# ============================================================
# Feature engineering
# ============================================================

def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_step_turn(df: pd.DataFrame):
    df = df.copy()

    g = df.groupby("segment", sort=False)

    df["dt"] = g["t"].shift(-1) - df["t"]
    df["dx"] = g["x"].shift(-1) - df["x"]
    df["dy"] = g["y"].shift(-1) - df["y"]

    df["step_length"] = np.hypot(df["dx"], df["dy"])

    df["speed"] = df["step_length"] / df["dt"]
    df.loc[df["dt"] <= 0, "speed"] = np.nan

    heading = np.arctan2(df["dy"], df["dx"])
    next_heading = heading.groupby(df["segment"], sort=False).shift(-1)

    df["turn"] = wrap_pi(next_heading - heading)
    df["abs_turn"] = df["turn"].abs()

    return df


def compute_tortuosity(df: pd.DataFrame, half_window: int = 3):
    df = df.sort_values(["segment", "t"]).copy()
    g = df.groupby("segment", sort=False)

    if "step_length" not in df.columns:
        df["dx"] = g["x"].shift(-1) - df["x"]
        df["dy"] = g["y"].shift(-1) - df["y"]
        df["step_length"] = np.hypot(df["dx"], df["dy"]).astype(np.float32)

    window_steps = 2 * half_window

    df["path_length"] = (
        g["step_length"]
        .transform(lambda s: s.rolling(window_steps, center=True, min_periods=window_steps).sum())
        .astype(np.float32)
    )

    x_left = g["x"].shift(half_window)
    y_left = g["y"].shift(half_window)
    x_right = g["x"].shift(-half_window)
    y_right = g["y"].shift(-half_window)

    df["net_displacement"] = np.hypot(x_right - x_left, y_right - y_left).astype(np.float32)
    df["tortuosity"] = (df["path_length"] / df["net_displacement"]).astype(np.float32)
    df["straightness"] = (df["net_displacement"] / df["path_length"]).astype(np.float32)

    df.loc[~np.isfinite(df["tortuosity"]), "tortuosity"] = np.nan
    df.loc[~np.isfinite(df["straightness"]), "straightness"] = np.nan

    return df


# ============================================================
# State inference
# ============================================================

def cluster_to_state(df: pd.DataFrame):
    df = df.copy()

    centers = (
        df.dropna(subset=["cluster", "abs_turn"])
        .groupby("cluster")[["abs_turn"]]
        .mean()
    )

    score = centers["abs_turn"]
    exploit_cluster = score.idxmax()
    explore_cluster = [c for c in centers.index if c != exploit_cluster][0]

    df["state"] = pd.Series(index=df.index, dtype="object")
    df.loc[df["cluster"] == explore_cluster, "state"] = "explore"
    df.loc[df["cluster"] == exploit_cluster, "state"] = "exploit"

    return df


def determine_state_kmeans(df: pd.DataFrame, n_clusters: int = 2, random_state: int = 0):
    df = df.copy()

    features = df[["speed", "abs_turn", "tortuosity"]].dropna()
    X = features[["speed", "abs_turn", "tortuosity"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)

    df.loc[features.index, "cluster"] = labels
    df = cluster_to_state(df)

    return df


# ============================================================
# Fitting helpers
# ============================================================

def fit_crw_speed(df: pd.DataFrame):
    d = df[["segment", "t", "speed"]].dropna().copy()
    d = d.sort_values(["segment", "t"])

    g = d.groupby("segment", sort=False)
    d["speed_next"] = g["speed"].shift(-1)
    d = d.dropna(subset=["speed", "speed_next"])

    x = d["speed"].to_numpy()
    y = d["speed_next"].to_numpy()

    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, phi = coef

    phi = float(phi)
    c = float(c)

    speed_smooth = 1.0 - phi
    target_speed = float(np.mean(y)) if speed_smooth <= 1e-8 else float(c / speed_smooth)

    resid = y - (c + phi * x)
    speed_sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    return {
        "target_speed": target_speed,
        "speed_smooth": float(speed_smooth),
        "speed_sigma": speed_sigma,
    }


def _simulate_dir_steps(persistence: float, turn_sigma: float, n: int = 50_000, random_state: int = 0):
    """
    Exact 2D one-step turn distribution for:
        new_dir = unit(persistence * old_dir + noise)
        noise ~ N(0, turn_sigma^2 I)
    with old_dir aligned to +x.
    """
    rng = np.random.default_rng(random_state)
    eps = rng.normal(0.0, turn_sigma, size=(n, 2))
    proposal_x = persistence + eps[:, 0]
    proposal_y = eps[:, 1]
    return np.arctan2(proposal_y, proposal_x)


def turn_score(obs, sim_turn, bins: int = 72):
    obs = wrap_pi(np.asarray(obs))
    sim_turn = wrap_pi(np.asarray(sim_turn))

    h_obs, _ = np.histogram(obs, bins=bins, range=(-np.pi, np.pi), density=False)
    h_sim, _ = np.histogram(sim_turn, bins=bins, range=(-np.pi, np.pi), density=False)

    p_obs = h_obs.astype(float)
    p_sim = h_sim.astype(float)

    p_obs /= p_obs.sum()
    p_sim /= p_sim.sum()

    return np.mean((p_obs - p_sim) ** 2)


def fit_crw_turn(
    df: pd.DataFrame,
    persistence_grid=None,
    turn_sigma_grid=None,
    n_sim: int = 50_000,
    random_state: int = 0,
):
    obs = df["turn"].dropna().to_numpy()
    obs = obs[np.isfinite(obs)]

    if len(obs) == 0:
        return {
            "persistence": 0.0,
            "turn_sigma": 1.0,
        }

    if persistence_grid is None:
        persistence_grid = np.linspace(0.0, 2.0, 20)

    if turn_sigma_grid is None:
        turn_sigma_grid = np.linspace(0.05, 1.0, 10)

    best = None
    best_score = np.inf

    for persistence in persistence_grid:
        for turn_sigma in turn_sigma_grid:
            sim_turn = _simulate_dir_steps(
                persistence=persistence,
                turn_sigma=turn_sigma,
                n=n_sim,
                random_state=random_state,
            )
            score = turn_score(obs, sim_turn)

            if score < best_score:
                best_score = score
                best = {
                    "persistence": float(persistence),
                    "turn_sigma": float(turn_sigma),
                }

    return best


def fit_CRW(df: pd.DataFrame):
    speed_params = fit_crw_speed(df)
    turn_params = fit_crw_turn(df)
    return CRW_CFG(**speed_params, **turn_params, bias_gain=0.0)


def fit_ttl_trigger(df: pd.DataFrame, explore_label: str = "explore", exploit_label: str = "exploit"):
    d = df[["segment", "t", "state"]].dropna().sort_values(["segment", "t"]).copy()

    d["dt"] = d.groupby("segment", sort=False)["t"].shift(-1) - d["t"]

    # fill last dt in each segment with segment median dt
    seg_med_dt = d.groupby("segment", sort=False)["dt"].transform("median")
    d["dt"] = d["dt"].fillna(seg_med_dt)

    d["run_id"] = (
        d["segment"].ne(d["segment"].shift()) |
        d["state"].ne(d["state"].shift())
    ).cumsum()

    runs = (
        d.groupby("run_id", sort=False)
        .agg(
            segment=("segment", "first"),
            state=("state", "first"),
            t_start=("t", "first"),
            t_end=("t", "last"),
            n_points=("t", "size"),
            duration=("dt", "sum"),
        )
    )

    exploit_runs = runs[runs["state"].eq(exploit_label)].copy()
    dwell_times = exploit_runs["duration"].to_numpy()
    time_to_leave = float(np.median(dwell_times)) if len(dwell_times) else 0.0

    n_entries = int(len(exploit_runs))
    n_explore_steps = int(d["state"].eq(explore_label).sum())
    p_enter_per_step = (n_entries / n_explore_steps) if n_explore_steps > 0 else 0.0

    explore_runs = runs[runs["state"].eq(explore_label)].copy()
    explore_time = float(explore_runs["duration"].sum())
    lambda_enter = (n_entries / explore_time) if explore_time > 0 else 0.0

    return {
        "time_to_leave": time_to_leave,
        "p_enter_per_step": float(p_enter_per_step),
        "lambda_enter": float(lambda_enter),
        "n_entries": n_entries,
        "n_explore_steps": n_explore_steps,
    }


def fit_EE(df: pd.DataFrame, random_state: int = 0):
    labeled = determine_state_kmeans(df, n_clusters=2, random_state=random_state)

    df_explore = labeled[labeled["state"] == "explore"].copy()
    df_exploit = labeled[labeled["state"] == "exploit"].copy()

    explore_cfg = CRW_CFG(
        **fit_crw_speed(df_explore),
        **fit_crw_turn(df_explore),
        bias_gain=0.0,
    )
    exploit_cfg = CRW_CFG(
        **fit_crw_speed(df_exploit),
        **fit_crw_turn(df_exploit),
        bias_gain=0.0,
    )

    ttl_trigger = fit_ttl_trigger(labeled)

    ee_cfg = EE_CFG(
        explore_cfg=explore_cfg,
        exploit_cfg=exploit_cfg,
        time_to_leave=ttl_trigger["time_to_leave"],
    )

    return ee_cfg, ttl_trigger, labeled


def infer_pois_from_exploit(df: pd.DataFrame, eps: float = 25.0, min_samples: int = 20):
    pts = df[df["state"] == "exploit"][["x", "y"]].dropna().copy()
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    X = pts.to_numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    pts["poi_cluster"] = labels

    centers = (
        pts[pts["poi_cluster"] != -1]
        .groupby("poi_cluster")[["x", "y"]]
        .mean()
    )

    if len(centers) == 0:
        return pts[["x", "y"]].mean().to_numpy()[None, :]

    return centers.to_numpy()

def infer_pois_by_segment(df, eps=25.0, min_samples=20):
    poi_dict = {}

    exploit = df[df["state"] == "exploit"].dropna(subset=["x", "y"]).copy()

    for seg, part in exploit.groupby("segment", sort=False):
        if len(part) < min_samples:
            continue

        X = part[["x", "y"]].to_numpy()
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        part = part.copy()
        part["poi_cluster"] = labels

        centers = (
            part[part["poi_cluster"] != -1]
            .groupby("poi_cluster")[["x", "y"]]
            .mean()
        )

        if len(centers) > 0:
            poi_dict[int(seg)] = centers.to_numpy()
        else:
            # fallback: single centroid if exploit points exist but no DBSCAN cluster survives
            poi_dict[int(seg)] = part[["x", "y"]].mean().to_numpy()[None, :]

    return poi_dict


def fit_POI(
    ee_fit: tuple,
    random_state: int = 0,
    bias_gain: float = 0.5,
    arrive_dist: float = 10.0,
    poi_eps: float = 25.0,
    poi_min_samples: int = 20,
):
    ee_cfg, ttl_trigger, labeled = ee_fit
    pois = infer_pois_from_exploit(labeled, eps=poi_eps, min_samples=poi_min_samples)

    poi_cfg = POI_CFG(
        explore_cfg=CRW_CFG(
            persistence=ee_cfg.explore_cfg.persistence,
            turn_sigma=ee_cfg.explore_cfg.turn_sigma,
            target_speed=ee_cfg.explore_cfg.target_speed,
            speed_sigma=ee_cfg.explore_cfg.speed_sigma,
            speed_smooth=ee_cfg.explore_cfg.speed_smooth,
            bias_gain=bias_gain,
        ),
        exploit_cfg=ee_cfg.exploit_cfg,
        arrive_dist=arrive_dist,
        time_to_leave=ee_cfg.time_to_leave,
    )

    return poi_cfg, pois, labeled, ttl_trigger

def infer_poi_weights(
    df: pd.DataFrame,
    pois,
    radius: float = 10.0,
    min_return_time: float = 0.0,
    alpha: float = 0.25,
    beta: float = 0.01,
):
    pois = np.asarray(pois, dtype=float)
    n_poi = len(pois)

    if n_poi == 0:
        return np.zeros(0, dtype=float)

    entries = np.zeros(n_poi, dtype=float)
    revisits = np.zeros(n_poi, dtype=float)
    dwell = np.zeros(n_poi, dtype=float)

    use = df.dropna(subset=["x", "y", "t"]).sort_values(["segment", "t"]).copy()

    for _, part in use.groupby("segment", sort=False):
        xy = part[["x", "y"]].to_numpy(dtype=float)
        t = part["t"].to_numpy(dtype=float)

        d = np.hypot(
            xy[:, None, 0] - pois[None, :, 0],
            xy[:, None, 1] - pois[None, :, 1],
        )
        pid = d.argmin(axis=1)
        mind = d[np.arange(len(xy)), pid]
        pid[mind > radius] = -1

        # dwell time
        if len(t) > 1:
            dt = np.diff(t, append=t[-1])
        else:
            dt = np.array([0.0])

        for i in range(n_poi):
            dwell[i] += dt[pid == i].sum()

        # episode entries
        is_entry = (pid != -1) & np.r_[True, pid[1:] != pid[:-1]]
        entry_pid = pid[is_entry].astype(int)
        entry_t = t[is_entry]

        last_entry_t = {}
        for p, tt in zip(entry_pid, entry_t):
            entries[p] += 1
            if p in last_entry_t and (tt - last_entry_t[p]) >= min_return_time:
                revisits[p] += 1
            last_entry_t[p] = float(tt)

    raw = revisits + alpha * entries + beta * dwell

    if raw.sum() <= 0:
        return np.full(n_poi, 1.0 / n_poi)

    return raw / raw.sum()

def fit_LPOI(
    ee_fit: tuple,
    random_state: int = 0,
    bias_gain: float = 0.5,
    arrive_dist: float = 10.0,
    poi_eps: float = 25.0,
    poi_min_samples: int = 20,
    epsilon: float = 0.2,
    learning_rate: float = 0.3,
    disturbance_penalty: float = 1.0,
):
    poi_cfg, pois, labeled, _ = fit_POI(
        ee_fit=ee_fit,
        random_state=random_state,
        bias_gain=bias_gain,
        arrive_dist=arrive_dist,
        poi_eps=poi_eps,
        poi_min_samples=poi_min_samples,
    )

    poi_weights = infer_poi_weights(
        labeled[labeled["state"] == "exploit"],
        pois=pois,
        radius=arrive_dist,
        min_return_time=3.0,
    )

    lpoi_cfg = LPOI_CFG(
        explore_cfg=poi_cfg.explore_cfg,
        exploit_cfg=poi_cfg.exploit_cfg,
        arrive_dist=poi_cfg.arrive_dist,
        time_to_leave=poi_cfg.time_to_leave,
        epsilon=epsilon,
        learning_rate=learning_rate,
        disturbance_penalty=disturbance_penalty,
    )

    return lpoi_cfg, pois, poi_weights, labeled


# ============================================================
# Dummy resource maps / simulation
# ============================================================

class NullMap:
    def is_encounter(self, pos, rng):
        return False, None

    def get_pois(self):
        return []


class RandomEncounterMap:
    def __init__(self, lambda_enter: float = 1.0, dt: float = 5.0):
        self.lambda_enter = lambda_enter
        self.dt = dt
        self.p_encounter = 1.0 - np.exp(-self.lambda_enter * self.dt)

    def is_encounter(self, pos, rng):
        return (rng.random() < self.p_encounter), None

    def get_pois(self):
        return []


class FixedPOIMap:
    def __init__(self, pois, encounter_radius=10.0):
        self.pois = np.asarray(pois, dtype=float)
        self.encounter_radius = encounter_radius

    def get_pois(self):
        return self.pois

    def is_encounter(self, pos, rng):
        if len(self.pois) == 0:
            return False, None
        d = np.hypot(self.pois[:, 0] - pos.x, self.pois[:, 1] - pos.y)
        return bool(np.any(d <= self.encounter_radius)), None


class DummyAnimal:
    def __init__(self, x=0.0, y=0.0, z=0.0, movement_dim=MovementDim.TWO_D, resource_map=None):
        self.pos = Vector(x, y, z)
        self.vel_dir = Vector(1.0, 0.0, 0.0)
        self.vel_speed = 0.0
        self.movement_dim = movement_dim
        self.resource_map = resource_map if resource_map is not None else NullMap()
        self.disturbance = 0.0


def update_animal_position(animal, dt):
    animal.pos = animal.pos.add(animal.vel_dir.scale(animal.vel_speed * dt))


def make_random_encounter_factory(lambda_enter: float, dt: float):
    def factory(seed):
        return RandomEncounterMap(lambda_enter=lambda_enter, dt=dt)
    return factory


def make_fixed_poi_factory(pois, arrive_dist: float):
    pois = np.asarray(pois, dtype=float)

    def factory(seed):
        return FixedPOIMap(pois=pois, encounter_radius=arrive_dist)
    return factory


def simulate_behavior(
    behavior,
    n_steps: int = 2048,
    dt: float = 5.0,
    n_seeds: int = 10,
    x0: float = 0.0,
    y0: float = 0.0,
    resource_map_factory=None,
):
    state_lut = {
        BehaviorState.EXPLOIT: "exploit",
        BehaviorState.EXPLORE: "explore",
    }

    frames = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        resource_map = resource_map_factory(seed) if resource_map_factory is not None else NullMap()
        animal = DummyAnimal(x=x0, y=y0, resource_map=resource_map)

        if hasattr(behavior, "reset"):
            try:
                behavior.reset(animal=animal, rng=rng)
            except TypeError:
                behavior.reset()

        rows = [{
            "segment": seed,
            "t": 0.0,
            "x": animal.pos.x,
            "y": animal.pos.y,
            "speed": animal.vel_speed,
            "dir_x": animal.vel_dir.x,
            "dir_y": animal.vel_dir.y,
            "state": state_lut.get(behavior.state, str(behavior.state)),
        }]

        for step in range(n_steps):
            done = behavior.fn(animal, rng, dt)
            update_animal_position(animal, dt)

            rows.append({
                "segment": seed,
                "t": (step + 1) * dt,
                "x": animal.pos.x,
                "y": animal.pos.y,
                "speed": animal.vel_speed,
                "dir_x": animal.vel_dir.x,
                "dir_y": animal.vel_dir.y,
                "state": state_lut.get(behavior.state, str(behavior.state)),
            })

            if done is True:
                break

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Evaluation
# ============================================================

def compute_distribution_scores(obs_df: pd.DataFrame, sim_df: pd.DataFrame):
    cols = ["speed", "turn", "tortuosity"]
    scores = {}

    states = ["explore", "exploit"]

    for col in cols:
        dists = []

        # only do per-state scoring if both dfs actually have state
        if "state" in obs_df.columns and "state" in sim_df.columns:
            for state in states:
                x = obs_df.loc[obs_df["state"] == state, col].dropna().to_numpy()
                y = sim_df.loc[sim_df["state"] == state, col].dropna().to_numpy()

                if len(x) > 0 and len(y) > 0:
                    wd = wasserstein_distance(x, y)
                    scores[f"{col}_{state}"] = float(wd)
                    dists.append(wd)

        # mean over available states
        if dists:
            scores[col] = float(np.mean(dists))
        else:
            # fallback: overall score when state is missing or no overlap
            x = obs_df[col].dropna().to_numpy()
            y = sim_df[col].dropna().to_numpy()
            if len(x) > 0 and len(y) > 0:
                scores[col] = float(wasserstein_distance(x, y))

    return scores

def compute_poi_revisitation_score(
    df: pd.DataFrame,
    pois,
    radius: float = 10.0,
    min_return_time: float = 0.0,
):
    pois = np.asarray(pois, dtype=float)
    if pois.size == 0:
        return {
            "revisitation_score": np.nan,
            "repeat_entries": 0,
            "total_entries": 0,
            "unique_pois_visited": 0,
        }

    total_entries = 0
    repeat_entries = 0
    unique_pois_visited = set()

    for _, part in df.sort_values(["segment", "t"]).groupby("segment", sort=False):
        xy = part[["x", "y"]].to_numpy(dtype=float)
        t = part["t"].to_numpy(dtype=float)
        if len(xy) == 0:
            continue

        # nearest POI assignment, masked outside radius
        d = np.hypot(
            xy[:, None, 0] - pois[None, :, 0],
            xy[:, None, 1] - pois[None, :, 1],
        )
        poi_id = d.argmin(axis=1)
        poi_dist = d[np.arange(len(xy)), poi_id]
        poi_id[poi_dist > radius] = -1

        # start of each POI visit episode
        is_entry = (poi_id != -1) & np.r_[True, poi_id[1:] != poi_id[:-1]]
        entry_pois = poi_id[is_entry].astype(int)
        entry_t = t[is_entry]

        last_entry_t = {}
        for pid, tt in zip(entry_pois, entry_t):
            total_entries += 1
            unique_pois_visited.add(int(pid))

            last_t = last_entry_t.get(int(pid))
            if last_t is not None and (tt - last_t) >= min_return_time:
                repeat_entries += 1

            last_entry_t[int(pid)] = float(tt)

    revisitation_score = repeat_entries / total_entries if total_entries > 0 else np.nan

    return {
        "revisitation_score": float(revisitation_score) if np.isfinite(revisitation_score) else 0,
        "repeat_entries": int(repeat_entries),
        "total_entries": int(total_entries),
        "unique_pois_visited": int(len(unique_pois_visited)),
    }

def evaluate_fit_distribution(
    behavior,
    df: pd.DataFrame,
    n_steps: int = 204_800,
    dt: float = 5.0,
    n_seeds: int = 5,
    resource_map_factory=None,
):
    sim_df = simulate_behavior(
        behavior,
        n_steps=n_steps,
        dt=dt,
        n_seeds=n_seeds,
        resource_map_factory=resource_map_factory,
        x0 = float(df["x"].mean()),
        y0 = float(df["y"].mean()),
    )

    sim_df = compute_step_turn(sim_df)
    sim_df = compute_tortuosity(sim_df, half_window=3)

    return compute_distribution_scores(df, sim_df), sim_df


# ============================================================
# Plotting
# ============================================================

def _ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_speed_turn_hist(df: pd.DataFrame, out: str | Path):
    _ensure_parent(out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for state in sorted(df["state"].dropna().unique()):
        mask = df["state"] == state
        axes[0].hist(df.loc[mask, "speed"].dropna(), bins=50, alpha=0.5, label=str(state))
        axes[1].hist(df.loc[mask, "turn"].dropna(), bins=50, alpha=0.5, label=str(state))

    axes[0].set_xlabel("speed")
    axes[0].set_ylabel("count")
    axes[0].set_title("Speed by state")
    axes[0].legend()

    axes[1].set_xlabel("turn (radians)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Turn by state")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_states(df: pd.DataFrame, out: str | Path):
    _ensure_parent(out)

    plt.figure()
    for state in sorted(df["state"].dropna().unique()):
        mask = df["state"] == state
        plt.scatter(df.loc[mask, "x"], df.loc[mask, "y"], alpha=0.2, s=5, label=str(state))

    plt.legend()
    plt.title("States")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=20_480)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--poi_bias_gain", type=float, default=0.5)
    parser.add_argument("--poi_arrive_dist", type=float, default=10.0)
    parser.add_argument("--poi_eps", type=float, default=25.0)
    parser.add_argument("--poi_min_samples", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, df = load_segments_df(args.manifest)
    df = compute_step_turn(df)
    df = compute_tortuosity(df, half_window=3)

    dt = 1.0
    print("mean dt:", dt)

    labeled = determine_state_kmeans(df, random_state=args.random_state)
    plot_speed_turn_hist(labeled, outdir / "speed_turn_hist_real.png")
    plot_scatter_states(labeled, outdir / "scatter_real.png")

    # CRW
    crw_cfg = fit_CRW(df)
    # print(crw_cfg)
    crw_behavior = CorrelatedRandomWalk(crw_cfg)
    crw_scores, crw_sim = evaluate_fit_distribution(
        crw_behavior,
        df,
        n_steps=args.n_steps,
        dt=dt,
        n_seeds=args.n_seeds,
    )
    plot_speed_turn_hist(crw_sim, outdir / "speed_turn_hist_crw_sim.png")

    # EE
    ee_fit = fit_EE(df, random_state=args.random_state)
    ee_cfg, ttl_trigger, ee_labeled = ee_fit
    # print(ee_cfg)
    ee_behavior = ExploreExploit(ee_cfg)
    ee_scores, ee_sim = evaluate_fit_distribution(
        ee_behavior,
        labeled,
        n_steps=args.n_steps,
        dt=dt,
        n_seeds=args.n_seeds,
        resource_map_factory=make_random_encounter_factory(
            lambda_enter=ttl_trigger["lambda_enter"],
            dt=dt,
        ),
    )
    plot_speed_turn_hist(ee_sim, outdir / "speed_turn_hist_ee_sim.png")

    # POI
    poi_cfg, pois, poi_labeled, _ = fit_POI(
        ee_fit,
        random_state=args.random_state,
        bias_gain=args.poi_bias_gain,
        arrive_dist=args.poi_arrive_dist,
        poi_eps=args.poi_eps,
        poi_min_samples=args.poi_min_samples,
    )
    # print(poi_cfg)
    poi_behavior = PointOfInterest(poi_cfg)
    poi_scores, poi_sim = evaluate_fit_distribution(
        poi_behavior,
        labeled,
        n_steps=args.n_steps,
        dt=dt,
        n_seeds=args.n_seeds,
        resource_map_factory=make_fixed_poi_factory(
            pois=pois,
            arrive_dist=poi_cfg.arrive_dist,
        ),
    )
    plot_speed_turn_hist(poi_sim, outdir / "speed_turn_hist_poi_sim.png")
    
    # LPOI
    lpoi_cfg, lpois, poi_weights, lpoi_labeled = fit_LPOI(
        ee_fit,
        random_state=args.random_state,
        bias_gain=args.poi_bias_gain,
        arrive_dist=args.poi_arrive_dist,
        poi_eps=args.poi_eps,
        poi_min_samples=args.poi_min_samples,
    )
    # print(lpoi_cfg)
    lpoi_behavior = LearningPointOfInterest(lpoi_cfg, reset_memory=False)
    lpoi_behavior.poi_values = {
        tuple(map(float, poi)): float(w)
        for poi, w in zip(lpois, poi_weights)
    }
    lpoi_scores, lpoi_sim = evaluate_fit_distribution(
        lpoi_behavior,
        labeled,
        n_steps=args.n_steps,
        dt=dt,
        n_seeds=args.n_seeds,
        resource_map_factory=make_fixed_poi_factory(
            pois=lpois,
            arrive_dist=lpoi_cfg.arrive_dist,
        ),
    )
    plot_speed_turn_hist(lpoi_sim, outdir / "speed_turn_hist_lpoi_sim.png")

    revisit_kwargs = {
        "pois": pois,
        "radius": args.poi_arrive_dist,
        "min_return_time": 3.0 * dt,   # filters boundary jitter / immediate re-entry
    }

    real_revisit = compute_poi_revisitation_score(labeled, **revisit_kwargs)
    crw_revisit = compute_poi_revisitation_score(crw_sim, **revisit_kwargs)
    ee_revisit = compute_poi_revisitation_score(ee_sim, **revisit_kwargs)
    poi_revisit = compute_poi_revisitation_score(poi_sim, **revisit_kwargs)
    lpoi_revisit = compute_poi_revisitation_score(lpoi_sim, **revisit_kwargs)

    crw_scores["revisitation"] = abs(crw_revisit["revisitation_score"] - real_revisit["revisitation_score"])
    ee_scores["revisitation"] = abs(ee_revisit["revisitation_score"] - real_revisit["revisitation_score"])
    poi_scores["revisitation"] = abs(poi_revisit["revisitation_score"] - real_revisit["revisitation_score"])
    lpoi_scores["revisitation"] = abs(lpoi_revisit["revisitation_score"] - real_revisit["revisitation_score"])
    with open(outdir / "report.txt", "w") as f:
        print(
            "REAL revisitation:",
            real_revisit["revisitation_score"],
            "repeat_entries:", real_revisit["repeat_entries"],
            "total_entries:", real_revisit["total_entries"], file=f)

        print(
            "CRW,  speed:", crw_scores["speed"],
            "turn:", crw_scores["turn"],
            "tortuosity:", crw_scores["tortuosity"],
            "sim_revisit:", crw_revisit["revisitation_score"],
            "mean_score:", (crw_scores["speed"] + crw_scores["turn"] + crw_scores["tortuosity"] + 1 - crw_revisit["revisitation_score"]) / 4, file=f)

        print(
            "EE,   speed:", ee_scores["speed"],
            "turn:", ee_scores["turn"],
            "tortuosity:", ee_scores["tortuosity"],
            "sim_revisit:", ee_revisit["revisitation_score"],
            "mean_score:", (ee_scores["speed"] + ee_scores["turn"] + ee_scores["tortuosity"] + 1 - ee_revisit["revisitation_score"]) / 4, file=f)
        print(
            "POI,  speed:", poi_scores["speed"],
            "turn:", poi_scores["turn"],
            "tortuosity:", poi_scores["tortuosity"],
            "sim_revisit:", poi_revisit["revisitation_score"],
            "mean_score:", (poi_scores["speed"] + poi_scores["turn"] + poi_scores["tortuosity"] + 1 - poi_revisit["revisitation_score"]) / 4, file=f)
        print(
            "LPOI, speed:", lpoi_scores["speed"],
            "turn:", lpoi_scores["turn"],
            "tortuosity:", lpoi_scores["tortuosity"],
            "sim_revisit:", lpoi_revisit["revisitation_score"],
            "mean_score:", (lpoi_scores["speed"] + lpoi_scores["turn"] + lpoi_scores["tortuosity"] + 1 - lpoi_revisit["revisitation_score"]) / 4, file=f)
        print("n inferred POIs:", len(pois))

    save_configs_yaml(
        outdir / "fitted_configs.yaml",
        CRW_CFG=crw_cfg,
        EE_CFG=ee_cfg,
        POI_CFG=poi_cfg,
        LPOI_CFG=lpoi_cfg,
        REPLAY_CFG={
            "manifest_path": args.manifest,
            "selection": "random",
            "zero_centered": False,
        },
    )

if __name__ == "__main__":
    main()