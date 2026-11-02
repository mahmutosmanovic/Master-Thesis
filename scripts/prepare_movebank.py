"""Prepare Movebank CSV fixes as timed, local-metre REPLAY_CFG episodes.

No smoothing, spatial scaling, or synthetic turns are applied. Between fixes,
Replay uses linear interpolation. time_scale=1 preserves recorded speeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


def prepare(csv_path, out_dir, individual=None, max_gap=7200.0,
            episode_seconds=204.8, time_scale=1.0):
    checks = [("max_gap", max_gap), ("time_scale", time_scale)]
    if episode_seconds is not None:
        checks.append(("episode_seconds", episode_seconds))
    for name, value in checks:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {out_dir}")
    df = pd.read_csv(csv_path, dtype={"individual-local-identifier": str,
                                    "tag-local-identifier": str})
    original_rows = len(df)
    id_col = ("individual-local-identifier" if "individual-local-identifier" in df
              else "tag-local-identifier")
    required = [id_col, "timestamp", "location-long", "location-lat"]
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("location-long", "location-lat"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df["location-long"].between(-180, 180) & df["location-lat"].between(-80, 84)]
    if "visible" in df:
        df = df[df["visible"].astype(str).str.lower().isin(["true", "1"])]
    if "manually-marked-outlier" in df:
        df = df[~df["manually-marked-outlier"].astype(str).str.lower().isin(["true", "1"])]
    if individual is not None:
        df = df[df[id_col] == str(individual)]
    df = df.sort_values([id_col, "timestamp"], kind="stable")
    df = df.drop_duplicates([id_col, "timestamp"], keep="first")
    if len(df) < 2:
        raise ValueError("Fewer than two usable fixes; check filters and individual ID")

    rows, all_gaps, all_speeds, plot_tracks = [], [], [], []
    for identifier, animal in df.groupby(id_col, sort=False):
        gaps = animal["timestamp"].diff().dt.total_seconds()
        all_gaps.extend(gaps.dropna().tolist())
        for _, segment in animal.groupby(gaps.gt(max_gap).cumsum(), sort=False):
            if len(segment) < 2:
                continue
            lon = segment["location-long"].to_numpy()
            lat = segment["location-lat"].to_numpy()
            zone = min(60, max(1, int((np.median(lon) + 180) // 6) + 1))
            epsg = (32600 if np.median(lat) >= 0 else 32700) + zone
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            x, y = transformer.transform(lon, lat)
            # Subtract origin in float64 before saving, preserving metre precision.
            x, y = x - x[0], y - y[0]
            raw_t = (segment["timestamp"] - segment["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
            t = raw_t / time_scale
            all_speeds.extend((np.hypot(np.diff(x), np.diff(y)) / np.diff(raw_t)).tolist())
            plot_tracks.append((identifier, x, y))
            # Include exact window endpoints even when GPS fixes are hours apart.
            # This avoids discarding sparse but valid tracks or padding stationary tails.
            # None keeps one sparse full segment for on-demand episode sampling.
            window_seconds = t[-1] if episode_seconds is None else episode_seconds
            n_windows = int(np.ceil(t[-1] / window_seconds))
            for window in range(n_windows):
                start = window * window_seconds
                end = min(start + window_seconds, t[-1])
                if end - start < 1e-9:
                    continue
                knots = np.r_[start, t[(t > start) & (t < end)], end]
                wx, wy = np.interp(knots, t, x), np.interp(knots, t, y)
                origin_x, origin_y = wx[0], wy[0]
                wx, wy = wx - origin_x, wy - origin_y
                filename = f"{len(rows):08d}.npz"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_dir / filename, t=knots-start, x=wx, y=wy)
                rows.append({"path": filename, id_col: identifier,
                             "n_points": len(knots), "duration_s": end-start,
                             "start_time": segment["timestamp"].iloc[0] + pd.to_timedelta(start*time_scale, unit="s"),
                             "end_time": segment["timestamp"].iloc[0] + pd.to_timedelta(end*time_scale, unit="s"),
                             "time_scale": time_scale, "utm_epsg": epsg,
                             "xmin": float(wx.min()), "xmax": float(wx.max()),
                             "ymin": float(wy.min()), "ymax": float(wy.max())})
    if not rows:
        raise ValueError("No replayable segments; inspect sampling gaps and --max-gap")
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    report = {"source": str(Path(csv_path).resolve()), "input_rows": original_rows,
              "usable_rows": len(df), "individuals": int(df[id_col].nunique()),
              ("segments" if episode_seconds is None else "episodes"): len(rows), "max_gap_seconds": max_gap,
              "episode_seconds": episode_seconds, "time_scale": time_scale,
              "median_fix_interval_seconds": float(np.median(all_gaps)),
              "median_recorded_step_speed_mps": float(np.median(all_speeds)),
              "max_recorded_step_speed_mps": float(np.max(all_speeds)),
              "interpolation": "linear; movement between GPS fixes is unobserved"}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for identifier, x, y in plot_tracks[:20]:
        ax.plot(x, y, ".-", markersize=2, linewidth=0.7, label=str(identifier))
    ax.set(xlabel="East from segment start (m)", ylabel="North from segment start (m)",
           title="Recorded GPS fixes and straight-line interpolation")
    ax.set_aspect("equal", adjustable="datalim")
    if len(plot_tracks) <= 10:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=160)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--individual", help="Individual ID, e.g. Z3864 (uses tag ID if individual column absent)")
    parser.add_argument("--max-gap", type=float, default=7200, help="Maximum original GPS interval in seconds")
    parser.add_argument("--episode-seconds", type=float, default=204.8, help="Simulation seconds per window")
    parser.add_argument("--time-scale", type=float, default=1, help="Recorded seconds per simulation second; default preserves speeds")
    args = parser.parse_args()
    report = prepare(args.csv, args.out, args.individual, args.max_gap,
                     args.episode_seconds, args.time_scale)
    print(json.dumps(report, indent=2))
    print(f"Replay manifest: {Path(args.out).resolve() / 'manifest.parquet'}")
    if report["median_fix_interval_seconds"] > args.episode_seconds * args.time_scale:
        print("NOTE: Typical GPS intervals exceed one episode; many windows contain only a straight-line portion.")
    if args.time_scale != 1:
        print(f"NOTE: Time compressed by {args.time_scale:g}; simulated speeds are multiplied by this factor.")


if __name__ == "__main__":
    main()
