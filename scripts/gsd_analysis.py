"""
Post-hoc GSD / pixels-on-target analysis of evaluation step logs.

Works on any eval directory (old or new runs): it recomputes, per step and drone,
the slant range to the drone's target animal, whether the target is in the camera
frustum, the ground sample distance (GSD) and optionally pixels on target.

Camera model (same as the environment's `camera` block):
    equidistant / f-theta lens, ifov = hor_angle[rad] / sensor_px
    GSD = slant_distance * ifov                      [m/px]
    quality cost = clip(GSD / gsd_max, 0, 1), and 1 when the target is not visible
gsd_max defaults to the run's `camera` block, else to the default drone
(M2: 140 deg, 200 m, 3840 px -> 12.73 cm/px). For M2-only runs the GSD quality
equals the logged r_dist, so no retraining is needed.

Usage:
    python -m scripts.gsd_analysis evals/<eval_dir> [more dirs ...] \
        [--body-length 0.7] [--sensor-px 3840] [--out table/gsd_summary.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_SENSOR_PX = 3840
DEFAULT_REF_HOR_ANGLE = 140.0
DEFAULT_REF_VIEW_RANGE = 200.0

STEP_COLS = ["episode", "step", "entity_type", "x", "y", "z", "view_x", "view_y", "view_z"]
EP_COLS = ["reward", "calm_frac", "avoid_frac", "flee_frac", "p_disturbance", "r_dist"]


def _load_config(eval_dir: Path) -> dict:
    with open(eval_dir / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def _gsd_max(cfg: dict, sensor_px_override=None) -> float:
    cam = cfg.get("camera") or {}
    sensor_px = sensor_px_override or int(cam.get("sensor_px", DEFAULT_SENSOR_PX))
    ref_angle = float(cam.get("ref_hor_angle", DEFAULT_REF_HOR_ANGLE))
    ref_range = float(cam.get("ref_view_range", DEFAULT_REF_VIEW_RANGE))
    return ref_range * np.deg2rad(ref_angle) / sensor_px


def _drone_params(cfg: dict, sensor_px_override=None) -> pd.DataFrame:
    cam = cfg.get("camera") or {}
    rows = []
    for name, d in cfg["drone"].items():
        sensor_px = sensor_px_override or int(d.get("sensor_px", cam.get("sensor_px", DEFAULT_SENSOR_PX)))
        rows.append({
            "entity_type": str(name),
            "hor_angle": float(d["hor_angle"]),
            "ver_angle": float(d["ver_angle"]),
            "view_range": float(d["view_range"]),
            "ifov": np.deg2rad(float(d["hor_angle"])) / sensor_px,
        })
    return pd.DataFrame(rows)


def _step_csvs(eval_dir: Path):
    """Step logs are the <name>.csv files that have a matching <name>_ep.csv."""
    for ep in sorted(eval_dir.glob("*_ep.csv")):
        step = ep.with_name(ep.name[: -len("_ep.csv")] + ".csv")
        if step.exists():
            yield step, ep


def _per_drone_frame(step_csv: Path, cfg: dict, sensor_px_override=None) -> pd.DataFrame:
    df = pd.read_csv(step_csv, usecols=lambda c: c in STEP_COLS)
    df["entity_type"] = df["entity_type"].astype(str)

    animals = df[df["entity_type"].str.lower() == "animal"].copy()
    drones = df[df["entity_type"].str.lower() != "animal"].copy()

    # Row order within a step matches env.animals / env.drones; drone i targets animal i % n_animals.
    animals["a_idx"] = animals.groupby(["episode", "step"]).cumcount()
    drones["d_idx"] = drones.groupby(["episode", "step"]).cumcount()
    n_animals = int(cfg["animal"]["env"]["count"])
    drones["a_idx"] = drones["d_idx"] % n_animals

    animals = animals.rename(columns={"x": "ax", "y": "ay", "z": "az"})[["episode", "step", "a_idx", "ax", "ay", "az"]]
    m = drones.merge(animals, on=["episode", "step", "a_idx"], how="inner")
    m = m.merge(_drone_params(cfg, sensor_px_override), on="entity_type", how="left")
    if m["ifov"].isna().any():
        missing = sorted(m.loc[m["ifov"].isna(), "entity_type"].unique())
        raise ValueError(f"{step_csv}: drone types {missing} not found in config")

    rel = m[["ax", "ay", "az"]].to_numpy(float) - m[["x", "y", "z"]].to_numpy(float)
    dist = np.linalg.norm(rel, axis=1)
    rel_u = rel / np.maximum(dist, 1e-8)[:, None]

    # Camera basis, identical to Env._camera_basis
    cx_axis = m[["view_x", "view_y", "view_z"]].to_numpy(float)
    cx_axis /= np.linalg.norm(cx_axis, axis=1, keepdims=True) + 1e-8
    cy_axis = np.cross(np.array([0.0, 0.0, 1.0]), cx_axis)
    cy_axis /= np.linalg.norm(cy_axis, axis=1, keepdims=True) + 1e-8
    cz_axis = np.cross(cx_axis, cy_axis)
    cz_axis /= np.linalg.norm(cz_axis, axis=1, keepdims=True) + 1e-8

    cx = np.einsum("ij,ij->i", rel_u, cx_axis)
    cy = np.einsum("ij,ij->i", rel_u, cy_axis)
    cz = np.einsum("ij,ij->i", rel_u, cz_axis)
    v_ang = np.arctan2(cz, cx)
    h_ang = np.arctan2(cy, cx)

    m["dist"] = dist
    m["visible"] = (
        (cx > 0.0)
        & (np.abs(v_ang) <= np.deg2rad(m["ver_angle"].to_numpy() / 2.0))
        & (np.abs(h_ang) <= np.deg2rad(m["hor_angle"].to_numpy() / 2.0))
        & (dist <= m["view_range"].to_numpy())
    )
    m["gsd"] = dist * m["ifov"].to_numpy()
    return m


def analyse_step_csv(step_csv: Path, ep_csv: Path, cfg: dict, body_length=None, sensor_px_override=None):
    m = _per_drone_frame(step_csv, cfg, sensor_px_override)
    gsd_max = _gsd_max(cfg, sensor_px_override)

    m["q_cost"] = np.where(m["visible"], np.clip(m["gsd"] / gsd_max, 0.0, 1.0), 1.0)
    vis = m[m["visible"]]

    def _pot(gsd):
        return body_length / gsd if body_length else np.nan

    base = {"eval_dir": step_csv.parent.name, "policy": step_csv.stem, "gsd_max_cm": gsd_max * 100}

    # Per drone type
    rows = []
    for dtype, g in m.groupby("entity_type"):
        gv = g[g["visible"]]
        rows.append({
            **base,
            "level": f"type:{dtype}",
            "n_drones": int(g.groupby(["episode", "step"]).size().max()),
            "visible_frac": g["visible"].mean(),
            "dist_mean_m": gv["dist"].mean(),
            "gsd_mean_cm": gv["gsd"].mean() * 100,
            "gsd_median_cm": gv["gsd"].median() * 100,
            "pot_mean_px": _pot(gv["gsd"]).mean() if body_length else np.nan,
            "r_quality_gsd": -g["q_cost"].mean(),
        })

    # Team level: best (smallest) GSD on each target among drones that see it
    m["gsd_vis"] = m["gsd"].where(m["visible"])
    team = m.groupby(["episode", "step", "a_idx"]).agg(
        any_visible=("visible", "max"),
        best_gsd=("gsd_vis", "min"),
    )
    step_quality = m.groupby(["episode", "step"])["q_cost"].mean()
    team_vis = team[team["any_visible"]]

    ep = pd.read_csv(ep_csv)
    ep_means = {f"ep_{c}": ep[c].mean() for c in EP_COLS if c in ep.columns}

    rows.append({
        **base,
        "level": "team",
        "n_drones": int(m.groupby(["episode", "step"]).size().max()),
        "visible_frac": team["any_visible"].mean(),
        "dist_mean_m": vis["dist"].mean(),
        "gsd_mean_cm": vis["gsd"].mean() * 100,
        "gsd_median_cm": vis["gsd"].median() * 100,
        "team_best_gsd_mean_cm": team_vis["best_gsd"].mean() * 100,
        "team_best_gsd_median_cm": team_vis["best_gsd"].median() * 100,
        "pot_mean_px": _pot(vis["gsd"]).mean() if body_length else np.nan,
        "team_best_pot_mean_px": _pot(team_vis["best_gsd"]).mean() if body_length else np.nan,
        "r_quality_gsd": -step_quality.mean(),
        **ep_means,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    parser.add_argument("--body-length", type=float, default=None, help="Target body length [m] for pixels on target")
    parser.add_argument("--sensor-px", type=int, default=None, help="Override horizontal sensor pixels for all drones")
    parser.add_argument("--out", type=Path, default=None, help="Combined summary CSV (default: per-dir gsd_summary.csv only)")
    args = parser.parse_args()

    all_rows = []
    for eval_dir in args.eval_dirs:
        cfg = _load_config(eval_dir)
        dir_rows = []
        for step_csv, ep_csv in _step_csvs(eval_dir):
            dir_rows += analyse_step_csv(step_csv, ep_csv, cfg, args.body_length, args.sensor_px)
        if not dir_rows:
            print(f"[WARN] no step logs found in {eval_dir}")
            continue
        pd.DataFrame(dir_rows).to_csv(eval_dir / "gsd_summary.csv", index=False)
        all_rows += dir_rows

    if not all_rows:
        return
    out = pd.DataFrame(all_rows)
    show = ["eval_dir", "policy", "level", "visible_frac", "dist_mean_m", "gsd_mean_cm",
            "team_best_gsd_mean_cm", "pot_mean_px", "team_best_pot_mean_px", "r_quality_gsd", "ep_r_dist"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(out[[c for c in show if c in out.columns]].to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"GSD_SUMMARY::{args.out}")


if __name__ == "__main__":
    main()
