from __future__ import annotations

import argparse
import yaml
import numbers
from pathlib import Path
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

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

# region IO

def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_segments_df(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
    base_dir = manifest_path.parent

    frames = []
    for _, row in manifest.iterrows():
        seg_path = base_dir / row["path"]
        with np.load(seg_path, allow_pickle=False) as z:
            t = z["t"].astype(np.float32)
            x = z["x"].astype(np.float32)
            y = z["y"].astype(np.float32)

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

def save_configs_yaml(path, **configs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: _to_builtin(cfg) for name, cfg in configs.items()}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

# endregion

# region Feature engineering

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

def compute_tortuosity(df: pd.DataFrame, half_window = 3):
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

# endregion

# region State inference

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

def determine_state_kmeans(df: pd.DataFrame, n_clusters = 2, random_state = 0):
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

def determine_state_kmeans_smooth(df: pd.DataFrame, n_clusters = 2, random_state = 0, smooth_window = 5):
    df = df.copy()

    features = df[["speed", "abs_turn", "tortuosity"]].dropna()
    X = features[["speed", "abs_turn", "tortuosity"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)

    df.loc[features.index, "cluster"] = labels
    df = cluster_to_state(df)

    mask = df["state"].notna()

    # Temporal majority vote
    if mask.any():
        is_exploit = (df.loc[mask, "state"] == "exploit").astype(float)
        smoothed = is_exploit.groupby(df.loc[mask, "segment"], sort=False).transform(
            lambda s: s.rolling(window=smooth_window, center=True, min_periods=1).median()
        )
        
        df.loc[mask, "state"] = np.where(smoothed >= 0.5, "exploit", "explore")

    return df

def determine_state_hmm(df: pd.DataFrame, n_clusters = 2, random_state = 0):
    df = df.copy()

    features_cols =["speed", "abs_turn", "tortuosity"]
    
    valid_mask = df[features_cols].notna().all(axis=1)
    valid_df = df[valid_mask].sort_values(["segment", "t"]).copy()

    if len(valid_df) == 0:
        df["cluster"] = np.nan
        df["state"] = np.nan
        return df

    X = valid_df[features_cols]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lengths = valid_df.groupby("segment", sort=False).size().tolist()

    model = hmm.GaussianHMM(
        n_components=n_clusters, 
        covariance_type="full", 
        random_state=random_state, 
        n_iter=100
    )
    
    model.fit(Xs, lengths)
    labels = model.predict(Xs, lengths)

    valid_df["cluster"] = labels
    df.loc[valid_df.index, "cluster"] = valid_df["cluster"]
    
    df = cluster_to_state(df)

    return df

# endregion

# region Fitting

def scale_speed_parameters(speed_smooth_obs, speed_sigma_obs, dt_obs, dt_sim=0.1):
    retention_obs = 1.0 - speed_smooth_obs
    retention_sim = retention_obs ** (dt_sim / dt_obs)
        
    speed_smooth_sim = 1.0 - retention_sim
    
    steady_state_variance = (speed_sigma_obs**2) / (1.0 - retention_obs**2)
    speed_sigma_sim = np.sqrt(steady_state_variance * (1.0 - retention_sim**2))
    
    return speed_smooth_sim, speed_sigma_sim

def bin_dt(d, n_bins=3):
    d = d.copy()
    fixed = d["dt"].max() - d["dt"].min() > 1

    if fixed or n_bins <= 1:
        dt_rep = float(d["dt"].median())
        d["dt_bin"] = dt_rep
        return d

    dt_min = float(d["dt"].min())
    dt_max = float(d["dt"].max())

    cats = pd.cut(
        d["dt"],
        bins=np.linspace(dt_min, dt_max, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )

    rep_by_bin = d.groupby(cats, observed=True)["dt"].median()

    d["dt_bin"] = cats.map(rep_by_bin).astype(float)
    return d

def fit_crw_speed(df: pd.DataFrame, dt_sim = 0.1):
    d = df[["segment", "t", "dt", "speed"]].dropna().copy()
    d = d.sort_values(["segment", "t"])

    g = d.groupby("segment", sort=False)
    d["speed_next"] = g["speed"].shift(-1)
    d = d.dropna(subset=["speed", "speed_next", "dt"]).copy()

    if len(d) == 0:
        return {"target_speed": 0.0, "speed_smooth": 1.0, "speed_sigma": 0.0}

    d = bin_dt(d, n_bins=3)
    bin_counts = d["dt_bin"].value_counts()
    valid_bins = bin_counts[bin_counts >= 30].index.to_list()
    
    # Only use dt bins that have enough points for a stable linear regression
    valid_bins = bin_counts[bin_counts >= 30].index.to_list()
    
    if not valid_bins:
        valid_bins = [float(d["dt"].median())]
        d["dt_bin"] = valid_bins[0]

    agg_smooth = 0.0
    agg_sigma = 0.0
    agg_target = 0.0
    total_weight = 0.0

    for dt_obs in valid_bins:
        subset = d[d["dt_bin"] == dt_obs]
        x = subset["speed"].to_numpy()
        y = subset["speed_next"].to_numpy()
        weight = len(x)
        
        # Fit the OU process for this specific observation interval
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c, phi = coef

        phi = float(phi)
        c = float(c)

        # Handle edge cases where phi might be > 1.0 due to noise
        speed_smooth_obs = max(0.0, min(1.0, 1.0 - phi))
        target_speed = float(np.mean(y)) if speed_smooth_obs <= 1e-8 else float(c / speed_smooth_obs)

        resid = y - (c + phi * x)
        speed_sigma_obs = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

        # Scale parameters from this specific dt_obs down to dt_sim (0.1)
        smooth_sim, sigma_sim = scale_speed_parameters(
            speed_smooth_obs, speed_sigma_obs, dt_obs, dt_sim
        )
        # smooth_sim, sigma_sim = speed_smooth_obs, speed_sigma_obs

        agg_smooth += smooth_sim * weight
        agg_sigma += sigma_sim * weight
        agg_target += target_speed * weight
        total_weight += weight

    return {
        "target_speed": float(agg_target / total_weight),
        "speed_smooth": float(agg_smooth / total_weight),
        "speed_sigma": float(agg_sigma / total_weight),
    }

def _simulate_dir_steps(persistence_sim, turn_sigma_sim, dt_obs, dt_sim=0.1, n=5_000, random_state=0):
    rng = np.random.default_rng(random_state)
    
    # How many RL steps happen between two GPS pings?
    n_microsteps = max(1, int(round(dt_obs / dt_sim)))
    
    # Start all 50,000 simulated animals pointing exactly at +x (1, 0)
    dirs = np.zeros((n, 2))
    dirs[:, 0] = 1.0 
    
    for _ in range(n_microsteps):
        # Generate noise for all n animals
        eps = rng.normal(0.0, turn_sigma_sim, size=(n, 2))
        
        # Apply the exact math from your step_direction kernel
        proposal_x = persistence_sim * dirs[:, 0] + eps[:, 0]
        proposal_y = persistence_sim * dirs[:, 1] + eps[:, 1]
        
        # .unit() normalization
        mags = np.hypot(proposal_x, proposal_y)
        dirs[:, 0] = proposal_x / mags
        dirs[:, 1] = proposal_y / mags

    # Return the final aggregated angle after n_microsteps
    return np.arctan2(dirs[:, 1], dirs[:, 0])

def turn_score(obs, sim_turn, bins = 72):
    obs = wrap_pi(np.asarray(obs))
    sim_turn = wrap_pi(np.asarray(sim_turn))

    h_obs, _ = np.histogram(obs, bins=bins, range=(-np.pi, np.pi), density=False)
    h_sim, _ = np.histogram(sim_turn, bins=bins, range=(-np.pi, np.pi), density=False)

    p_obs = h_obs.astype(float)
    p_sim = h_sim.astype(float)

    p_obs /= p_obs.sum()
    p_sim /= p_sim.sum()

    return np.mean((p_obs - p_sim) ** 2)

def circular_scale(obs):
    obs = wrap_pi(np.asarray(obs, dtype=float))
    obs = obs[np.isfinite(obs)]

    if len(obs) == 0:
        return 1.0

    # circular mean direction
    mu = np.angle(np.mean(np.exp(1j * obs)))

    # mean absolute circular deviation from the mean direction
    scale = np.mean(np.abs(wrap_pi(obs - mu)))

    if not np.isfinite(scale) or scale <= 1e-8:
        return 1.0

    return float(scale)

def circular_turn_distance(obs, sim_turn, bins=72, scale=None):
    obs = wrap_pi(np.asarray(obs, dtype=float))
    sim_turn = wrap_pi(np.asarray(sim_turn, dtype=float))

    obs = obs[np.isfinite(obs)]
    sim_turn = sim_turn[np.isfinite(sim_turn)]

    if len(obs) == 0 or len(sim_turn) == 0:
        return np.nan

    h_obs, _ = np.histogram(obs, bins=bins, range=(-np.pi, np.pi), density=False)
    h_sim, _ = np.histogram(sim_turn, bins=bins, range=(-np.pi, np.pi), density=False)

    p = h_obs.astype(float)
    q = h_sim.astype(float)

    p /= p.sum()
    q /= q.sum()

    s = np.cumsum(p - q)
    c = np.median(s)

    delta = 2.0 * np.pi / bins
    w1 = delta * np.sum(np.abs(s - c))

    if scale is None:
        scale = circular_scale(obs)

    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    return float(w1 / scale)

def fit_crw_turn(
    df: pd.DataFrame,
    dt_sim = 0.1,
    persistence_grid=None,
    turn_sigma_grid=None,
    n_sim = 10_000,
    random_state = 0,
):
    d = df[["turn", "dt"]].dropna()
    d = d[np.isfinite(d["turn"]) & (d["dt"] > 0)].copy()

    if len(d) == 0:
        return {
            "persistence": 0.0,
            "turn_sigma": 1.0,
        }

    # Bin dt to nearest 0.5s to handle variable sampling
    d["dt_bin"] = np.round(d["dt"] * 2) / 2
    
    bin_counts = d["dt_bin"].value_counts()
    valid_bins = bin_counts[bin_counts >= 50].index.to_list()
    
    if not valid_bins:
        valid_bins =[float(d["dt"].median())]
        d["dt_bin"] = valid_bins[0]

    # Because dt=0.1 is very fine, persistence is usually higher and sigma is lower
    if persistence_grid is None:
        persistence_grid = np.linspace(0.1, 1.2, 5) 
    if turn_sigma_grid is None:
        turn_sigma_grid = np.linspace(0.01, 0.8, 5)

    best = None
    best_score = np.inf

    for persistence in persistence_grid:
        for turn_sigma in turn_sigma_grid:
            total_score = 0.0
            total_weight = 0.0
            
            for dt_obs in valid_bins:
                obs = d.loc[d["dt_bin"] == dt_obs, "turn"].to_numpy()
                weight = len(obs)
                
                # Micro-step the proposed RL parameters to match this bin's observation gap
                sim_turn = _simulate_dir_steps(
                    persistence_sim=persistence,
                    turn_sigma_sim=turn_sigma,
                    dt_obs=dt_obs,
                    dt_sim=dt_sim,
                    n=n_sim,
                    random_state=random_state,
                )
                
                score = circular_turn_distance(obs, sim_turn)
                total_score += score * weight
                total_weight += weight

            avg_score = total_score / total_weight

            if avg_score < best_score:
                best_score = avg_score
                best = {
                    "persistence": float(persistence),
                    "turn_sigma": float(turn_sigma),
                }

    return best

def fit_CRW(df: pd.DataFrame, dt_sim = 0.1):
    speed_params = fit_crw_speed(df, dt_sim=dt_sim)
    turn_params = fit_crw_turn(df, dt_sim=dt_sim)
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


def fit_EE(df: pd.DataFrame, random_state = 0, dt_sim = 0.1):
    df_explore = df[df["state"] == "explore"].copy()
    df_exploit = df[df["state"] == "exploit"].copy()

    explore_cfg = CRW_CFG(
        **fit_crw_speed(df_explore, dt_sim=dt_sim),
        **fit_crw_turn(df_explore, dt_sim=dt_sim),
        bias_gain=0.0,
    )
    exploit_cfg = CRW_CFG(
        **fit_crw_speed(df_exploit, dt_sim=dt_sim),
        **fit_crw_turn(df_exploit, dt_sim=dt_sim),
        bias_gain=0.0,
    )

    ttl_trigger = fit_ttl_trigger(df)

    ee_cfg = EE_CFG(
        explore_cfg=explore_cfg,
        exploit_cfg=exploit_cfg,
        time_to_leave=ttl_trigger["time_to_leave"],
    )

    return ee_cfg, ttl_trigger, df

def infer_pois_from_exploit(df: pd.DataFrame, eps = 25.0, min_samples = 20):
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
    random_state = 0,
    bias_gain = 0.5,
    arrive_dist = 10.0,
    poi_eps = 25.0,
    poi_min_samples = 20,
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
    radius = 10.0,
    min_return_time = 0.0,
    alpha = 0.25,
    beta = 0.01,
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
    poi_fit: tuple,
    random_state = 0,
    bias_gain = 0.5,
    arrive_dist = 10.0,
    poi_eps = 25.0,
    poi_min_samples = 20,
    epsilon = 0.2,
    learning_rate = 0.3,
    disturbance_penalty = 1.0,
):
    poi_cfg, pois, labeled, _ = poi_fit

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

# endregion

# region Simulation / Dummy classes 

class NullMap:
    def is_encounter(self, pos, rng):
        return False, None

    def get_pois(self):
        return []

class RandomEncounterMap:
    def __init__(self, lambda_enter = 1.0, dt = 5.0):
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
    n_steps = 2048,
    dt = 5.0,
    n_seeds = 10,
    x0 = 0.0,
    y0 = 0.0,
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

# endregion

# region Evaluation

def _scale_std(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 1.0

    sd = float(np.std(x, ddof=1)) if len(x) > 1 else float(np.std(x))
    if sd > 0 and np.isfinite(sd):
        return sd

    return 1.0

def normalized_wasserstein(obs, sim, scale=None):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    obs = obs[np.isfinite(obs)]
    sim = sim[np.isfinite(sim)]

    if len(obs) == 0 or len(sim) == 0:
        return np.nan

    wd = float(wasserstein_distance(obs, sim))

    if scale is None:
        scale = _scale_std(obs)

    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    return wd / scale


def compute_distribution_scores(obs_df: pd.DataFrame, sim_df: pd.DataFrame):
    cols = ["speed", "turn", "tortuosity"]
    scores = {}
    states = ["explore", "exploit"]

    # normalize everything using the full observed distribution
    global_scales = {
        "speed": _scale_std(obs_df["speed"].to_numpy()),
        "tortuosity": _scale_std(obs_df["tortuosity"].to_numpy()),
        "turn": circular_scale(obs_df["turn"].to_numpy()),
    }

    have_states = (
        "state" in obs_df.columns
        and "state" in sim_df.columns
        and obs_df["state"].notna().any()
        and sim_df["state"].notna().any()
    )

    for col in cols:
        dists = []

        if have_states:
            for state in states:
                x = obs_df.loc[obs_df["state"] == state, col].dropna().to_numpy()
                y = sim_df.loc[sim_df["state"] == state, col].dropna().to_numpy()

                if len(x) > 0 and len(y) > 0:
                    if col == "turn":
                        dist = circular_turn_distance(
                            x, y, bins=72, scale=global_scales["turn"]
                        )
                    else:
                        dist = normalized_wasserstein(
                            x, y, scale=global_scales[col]
                        )

                    scores[f"{col}_{state}"] = float(dist)
                    dists.append(dist)

        if dists:
            scores[col] = float(np.mean(dists))
        else:
            x = obs_df[col].dropna().to_numpy()
            y = sim_df[col].dropna().to_numpy()

            if len(x) > 0 and len(y) > 0:
                if col == "turn":
                    dist = circular_turn_distance(
                        x, y, bins=72, scale=global_scales["turn"]
                    )
                else:
                    dist = normalized_wasserstein(
                        x, y, scale=global_scales[col]
                    )

                scores[col] = float(dist)

    return scores

def compute_poi_revisitation_score(
    df: pd.DataFrame,
    pois,
    radius = 10.0,
    min_return_time = 0.0,
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
    n_steps = 204_800,
    dt_sim = 0.1,
    n_seeds = 5,
    resource_map_factory=None,
):
    sim_df_high_res = simulate_behavior(
        behavior,
        n_steps=n_steps,
        dt=dt_sim,
        n_seeds=n_seeds,
        resource_map_factory=resource_map_factory,
        x0=float(df["x"].mean()),
        y0=float(df["y"].mean()),
    )
    
    valid_dts = df["dt"].dropna().to_numpy()
    valid_dts = valid_dts[valid_dts > 0]

    rng = np.random.default_rng(42)
    sampled_frames =[]

    # Subsample the simulation at gps rate
    for segment_id, group in sim_df_high_res.groupby("segment", sort=False):
        max_t = group["t"].max()

        sampled_dts = rng.choice(valid_dts, size=int((max_t / valid_dts.min()) + 100))
        obs_times = np.cumsum(np.insert(sampled_dts, 0, 0.0))
        obs_times = obs_times[obs_times <= max_t]

        indices = np.round(obs_times / dt_sim).astype(int)
        indices = np.clip(indices, 0, len(group) - 1)
        
        sampled_group = group.iloc[indices].copy()
        
        sampled_group["t"] = obs_times
        sampled_frames.append(sampled_group)

    sim_df = pd.concat(sampled_frames, ignore_index=True)
    sim_df = compute_step_turn(sim_df)
    sim_df = compute_tortuosity(sim_df, half_window=3)

    return compute_distribution_scores(df, sim_df), sim_df

# endregion

# region Plotting

def _state_color(state: str):
    return {
        "explore": "black",
        "exploit": "blue",
    }.get(str(state), "black")

def plot_speed_turn_hist(df: pd.DataFrame, out):
    _ensure_parent(out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for state in sorted(df["state"].dropna().unique()):
        mask = df["state"] == state
        color = _state_color(state)
        alpha = 0.4 if str(state) == "explore" else 0.5

        axes[0].hist(
            df.loc[mask, "speed"].dropna(),
            bins=50,
            alpha=alpha,
            label=str(state),
            color=color,
        )
        axes[1].hist(
            df.loc[mask, "turn"].dropna(),
            bins=50,
            alpha=alpha,
            label=str(state),
            color=color,
        )

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

def plot_scatter_states(df: pd.DataFrame, out):
    _ensure_parent(out)

    states = sorted(df["state"].dropna().unique())
    if len(states) == 0:
        return

    xmin, xmax = df["x"].min(), df["x"].max()
    ymin, ymax = df["y"].min(), df["y"].max()

    fig, axes = plt.subplots(1, len(states), figsize=(6 * len(states), 5), squeeze=False)
    axes = axes.ravel()

    for ax, state in zip(axes, states):
        mask = df["state"] == state
        ax.scatter(
            df.loc[mask, "x"],
            df.loc[mask, "y"],
            alpha=0.005,
            s=5,
            color=_state_color(state),
        )
        ax.set_title(f"State: {state}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_speed_turn_scatter(df: pd.DataFrame, out, n_sample: int = 5000):
    _ensure_parent(out)

    use = df.dropna(subset=["speed", "abs_turn", "state"]).copy()
    if len(use) > n_sample:
        use = use.sample(n_sample, random_state=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    for state in sorted(use["state"].unique()):
        part = use[use["state"] == state]
        ax.scatter(
            part["speed"],
            part["abs_turn"],
            s=6,
            alpha=0.3,
            label=state,
            color=_state_color(state),
        )

    ax.set_xlabel("speed")
    ax.set_ylabel("abs_turn")
    ax.set_title("Speed vs abs_turn")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_dt_hist(df: pd.DataFrame, out):
    _ensure_parent(out)

    vals = df["dt"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=50, color="black", alpha=0.5)
    ax.set_xlabel("dt")
    ax.set_ylabel("count")
    ax.set_title("Observation interval distribution")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_turn_polar(df: pd.DataFrame, out, bins: int = 36):
    _ensure_parent(out)

    states = sorted(df["state"].dropna().unique())
    if len(states) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(states),
        subplot_kw={"projection": "polar"},
        figsize=(5 * len(states), 4),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, state in zip(axes, states):
        vals = df.loc[df["state"] == state, "turn"].dropna().to_numpy()
        hist, edges = np.histogram(vals, bins=bins, range=(-np.pi, np.pi))
        theta = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges)
        ax.bar(
            theta,
            hist,
            width=width,
            alpha=0.5,
            color=_state_color(state),
        )
        ax.set_title(str(state))

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

# endregion

# region Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--poi_inference", type=str, default="km")
    parser.add_argument("--n_steps", type=int, default=500_000) 
    parser.add_argument("--dt_sim", type=float, default=0.1)
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

    dt_sim = args.dt_sim
    print(f"Targeting simulation tick rate: {dt_sim}s")

    match args.poi_inference:
        case "km":
            labeled = determine_state_kmeans(df, n_clusters=2, random_state=0)
        case "km_sm":
            labeled = determine_state_kmeans_smooth(df, n_clusters=2, random_state=0)
        case "hmm":
            labeled = determine_state_hmm(df, n_clusters=2, random_state=0)
        case _:
            raise NotImplementedError("Invalid poi_inference method")

    # CRW
    crw_cfg = fit_CRW(df, dt_sim=dt_sim)
    crw_behavior = CorrelatedRandomWalk(crw_cfg)
    crw_scores, crw_sim = evaluate_fit_distribution(
        crw_behavior,
        df,
        n_steps=args.n_steps,
        dt_sim=dt_sim,
        n_seeds=args.n_seeds,
    )

    # EE
    ee_fit = fit_EE(labeled, random_state=args.random_state, dt_sim=dt_sim)
    ee_cfg, ttl_trigger, ee_labeled = ee_fit
    ee_behavior = ExploreExploit(ee_cfg)
    ee_scores, ee_sim = evaluate_fit_distribution(
        ee_behavior,
        labeled,
        n_steps=args.n_steps,
        dt_sim=dt_sim,
        n_seeds=args.n_seeds,
        resource_map_factory=make_random_encounter_factory(
            lambda_enter=ttl_trigger["lambda_enter"],
            dt=dt_sim, # Passed dt_sim here so math uses 0.1s encounter probability
        ),
    )

    # POI
    poi_fit = fit_POI(
        ee_fit,
        random_state=args.random_state,
        bias_gain=args.poi_bias_gain,
        arrive_dist=args.poi_arrive_dist,
        poi_eps=args.poi_eps,
        poi_min_samples=args.poi_min_samples,
    )
    poi_cfg, pois, poi_labeled, _ = poi_fit
    poi_behavior = PointOfInterest(poi_cfg)
    poi_scores, poi_sim = evaluate_fit_distribution(
        poi_behavior,
        labeled,
        n_steps=args.n_steps,
        dt_sim=dt_sim,
        n_seeds=args.n_seeds,
        resource_map_factory=make_fixed_poi_factory(
            pois=pois,
            arrive_dist=poi_cfg.arrive_dist,
        ),
    )
    
    # LPOI
    lpoi_cfg, lpois, poi_weights, lpoi_labeled = fit_LPOI(
        poi_fit,
        random_state=args.random_state,
        bias_gain=args.poi_bias_gain,
        arrive_dist=args.poi_arrive_dist,
        poi_eps=args.poi_eps,
        poi_min_samples=args.poi_min_samples,
    )
    lpoi_behavior = LearningPointOfInterest(lpoi_cfg)
    lpoi_behavior.poi_values = {
        tuple(map(float, poi)): float(w)
        for poi, w in zip(lpois, poi_weights)
    }
    lpoi_scores, lpoi_sim = evaluate_fit_distribution(
        lpoi_behavior,
        labeled,
        n_steps=args.n_steps,
        dt_sim=dt_sim,
        n_seeds=args.n_seeds,
        resource_map_factory=make_fixed_poi_factory(
            pois=lpois,
            arrive_dist=lpoi_cfg.arrive_dist,
        ),
    )

    revisit_kwargs = {
        "pois": pois,
        "radius": args.poi_arrive_dist,
        "min_return_time": 30.0,
    }

    plot_speed_turn_hist(labeled, outdir / "real" / "speed_turn_hist_real.png")
    plot_scatter_states(labeled, outdir / "real" / "scatter_real.png")
    plot_speed_turn_scatter(labeled, outdir / "real" / "speed_turn_scatter_real.png")
    plot_dt_hist(labeled, outdir / "real" / "dt_hist_real.png")
    plot_turn_polar(labeled, outdir / "real" / "turn_polar_real.png")

    plot_speed_turn_hist(crw_sim, outdir / "crw" / "speed_turn_hist_crw_sim.png")
    plot_scatter_states(crw_sim, outdir / "crw" / "scatter_crw_sim.png")
    plot_speed_turn_scatter(crw_sim, outdir / "crw" / "speed_turn_scatter_crw_sim.png")
    plot_turn_polar(crw_sim, outdir / "crw" / "turn_polar_crw_sim.png")

    plot_speed_turn_hist(ee_sim, outdir / "ee" / "speed_turn_hist_ee_sim.png")
    plot_scatter_states(ee_sim, outdir / "ee" / "scatter_ee_sim.png")
    plot_speed_turn_scatter(ee_sim, outdir / "ee" / "speed_turn_scatter_ee_sim.png")
    plot_turn_polar(ee_sim, outdir / "ee" / "turn_polar_ee_sim.png")

    plot_speed_turn_hist(poi_sim, outdir / "poi" / "speed_turn_hist_poi_sim.png")
    plot_scatter_states(poi_sim, outdir / "poi" / "scatter_poi_sim.png")
    plot_speed_turn_scatter(poi_sim, outdir / "poi" / "speed_turn_scatter_poi_sim.png")
    plot_turn_polar(poi_sim, outdir / "poi" / "turn_polar_poi_sim.png")

    plot_speed_turn_hist(lpoi_sim, outdir / "lpoi" / "speed_turn_hist_lpoi_sim.png")
    plot_scatter_states(lpoi_sim, outdir / "lpoi" / "scatter_lpoi_sim.png")
    plot_speed_turn_scatter(lpoi_sim, outdir / "lpoi" / "speed_turn_scatter_lpoi_sim.png")
    plot_turn_polar(lpoi_sim, outdir / "lpoi" / "turn_polar_lpoi_sim.png")

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
            "revisit:", crw_scores["revisitation"],
            "mean_score:", (crw_scores["speed"] + crw_scores["turn"] + crw_scores["tortuosity"] + crw_scores["revisitation"]) / 4, file=f)

        print(
            "EE,   speed:", ee_scores["speed"],
            "turn:", ee_scores["turn"],
            "tortuosity:", ee_scores["tortuosity"],
            "revisit:", ee_scores["revisitation"],
            "mean_score:", (ee_scores["speed"] + ee_scores["turn"] + ee_scores["tortuosity"] + ee_scores["revisitation"]) / 4, file=f)
        print(
            "POI,  speed:", poi_scores["speed"],
            "turn:", poi_scores["turn"],
            "tortuosity:", poi_scores["tortuosity"],
            "revisit:", poi_scores["revisitation"],
            "mean_score:", (poi_scores["speed"] + poi_scores["turn"] + poi_scores["tortuosity"] + poi_scores["revisitation"]) / 4, file=f)
        print(
            "LPOI, speed:", lpoi_scores["speed"],
            "turn:", lpoi_scores["turn"],
            "tortuosity:", lpoi_scores["tortuosity"],
            "revisit:", lpoi_scores["revisitation"],
            "mean_score:", (lpoi_scores["speed"] + lpoi_scores["turn"] + lpoi_scores["tortuosity"] + lpoi_scores["revisitation"]) / 4, file=f)
        print("n inferred POIs:", len(pois))

    save_configs_yaml(
        outdir / "behaviors.yaml",
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

# endregion

if __name__ == "__main__":
    main()