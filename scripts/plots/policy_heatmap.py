import csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_positions_from_csv(csv_path):
    r_vals = []
    z_vals = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # keep drone rows only
            if row["entity_type"] == "animal":
                continue

            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])

            r = np.sqrt(x * x + y * y)

            r_vals.append(r)
            z_vals.append(z)

    if len(r_vals) == 0:
        raise ValueError(f"No drone rows found in {csv_path}")

    return np.array(r_vals), np.array(z_vals)


def make_output_path(csv_path):
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_policy_heatmap.png")


def plot_heatmap(r_vals, z_vals, out_path, bins=80):
    r_max = np.max(r_vals)
    z_max = np.max(z_vals)

    if r_max <= 0:
        r_max = 1.0
    if z_max <= 0:
        z_max = 1.0

    H, _, _ = np.histogram2d(
        r_vals,
        z_vals,
        bins=bins,
        range=[[0, r_max], [0, z_max]],
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, z_max],
    )

    plt.xlabel("Radial Distance (√(x²+y²))")
    plt.ylabel("Altitude (z)")
    plt.title("Drone Visitation Heatmap")

    cbar = plt.colorbar()
    cbar.set_label("Visit Count")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_policy_heatmap_from_csv(csv_path, bins=80):
    csv_path = Path(csv_path)
    r_vals, z_vals = load_positions_from_csv(csv_path)
    out_path = make_output_path(csv_path)
    plot_heatmap(r_vals, z_vals, out_path, bins=bins)
    return out_path

def plot_reward_heatmap_from_csv(
    csv_path,
    bins=80,
    cmap="YlGn",
    vmin=0.0,
    vmax=1.0,
    use_radial=True,
):
    """
    Plot a dense reward heatmap similar in style to the visitation heatmap.

    Color in each bin = mean step reward of drone positions in that bin.
    Unvisited bins are filled with vmin so the whole plot is rendered.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # keep drone rows only
    df = df[df["entity_type"] != "animal"].copy()

    # numeric conversion
    for col in ["x", "y", "z", "reward"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "reward"])

    if len(df) == 0:
        raise ValueError(f"No valid drone rows with reward found in {csv_path}")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    reward = np.clip(df["reward"].to_numpy(dtype=float), vmin, vmax)

    if use_radial:
        x_plot = np.sqrt(x**2 + y**2)
        y_plot = z
        xlabel = "Radial Distance (√(x²+y²))"
        ylabel = "Altitude (z)"
        title = "Mean Reward Heatmap (Radial Distance vs Altitude)"
        suffix = "_reward_heatmap_rz.png"
    else:
        x_plot = x
        y_plot = y
        xlabel = "x"
        ylabel = "y"
        title = "Mean Reward Heatmap (x vs y)"
        suffix = "_reward_heatmap_xy.png"

    x_max = np.max(x_plot)
    y_max = np.max(y_plot)

    if x_max <= 0:
        x_max = 1.0
    if y_max <= 0:
        y_max = 1.0

    # weighted reward sum
    reward_sum, _, _ = np.histogram2d(
        x_plot,
        y_plot,
        bins=bins,
        range=[[0, x_max], [0, y_max]],
        weights=reward,
    )

    # visit count
    counts, _, _ = np.histogram2d(
        x_plot,
        y_plot,
        bins=bins,
        range=[[0, x_max], [0, y_max]],
    )

    # dense matrix: unvisited bins take vmin so whole background is colored
    mean_reward = np.full_like(reward_sum, fill_value=vmin, dtype=float)
    visited = counts > 0
    mean_reward[visited] = reward_sum[visited] / counts[visited]

    plt.figure(figsize=(7, 6))

    plt.imshow(
        mean_reward.T,
        origin="lower",
        aspect="auto",
        extent=[0, x_max, 0, y_max],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_label("Mean Step Reward")

    out_path = csv_path.with_name(csv_path.stem + suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path

def _init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to step log CSV",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins",
    )

    return parser.parse_args()


def main():
    args = _init_argparse()

    out_path = make_output_path(args.csv)

    r_vals, z_vals = load_positions_from_csv(args.csv)
    plot_heatmap(r_vals, z_vals, out_path, bins=args.bins)

    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()