import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from environment import horizontal_gain, altitude_gain, angle_gain, truncate_colormap

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from matplotlib.patches import Polygon

def plot_visitation_on_disturbance_background(csv_path, bins=50, disturbance_cmap="Greys", title="Unspecified"):
    csv_path = Path(csv_path)
    r_vals, z_vals = load_positions_from_csv(csv_path)
    out_path = csv_path.with_name(f"{csv_path.stem}_policy_heatmap_disturbance_bg.png")

    r_vals = np.asarray(r_vals)
    z_vals = np.asarray(z_vals)

    r_max = 100
    z_max = 100

    # disturbance grid
    r_grid = np.linspace(0, r_max, bins)
    z_grid = np.linspace(0, z_max, bins)
    R, Z = np.meshgrid(r_grid, z_grid)

    G = np.zeros_like(R, dtype=float)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            dist_vec = (R[i, j], 0.0, Z[i, j])

            g_h = horizontal_gain(dist_vec)
            g_v = altitude_gain(dist_vec)
            g_a = angle_gain(dist_vec)

            G[i, j] = (g_h * g_v + g_a) / 2.0

    # normalize disturbance
    G = (G - G.min()) / (G.max() - G.min() + 1e-8)

    # visitation histogram
    H, _, _ = np.histogram2d(
        r_vals,
        z_vals,
        bins=bins,
        range=[[0, r_max], [0, z_max]],
    )
    H = H.T

    # normalize counts
    H_norm = H / (H.max() + 1e-8)

    # map counts -> Blues colormap
    visit_cmap = truncate_colormap(
        plt.get_cmap("Blues"),
        minval=0.3,
        maxval=1.0
    )

    blue_layer = visit_cmap(H_norm)

    # use visitation intensity as alpha
    blue_layer[..., 3] = H_norm


    fig, ax = plt.subplots(figsize=(7, 6))

    # disturbance background
    ax.imshow(
        G,
        origin="lower",
        extent=[0, r_max, 0, z_max],
        cmap=disturbance_cmap,
        aspect="auto",
        zorder=0,
    )

    # disturbance contour lines
    contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cs = ax.contour(
        G,
        levels=contour_levels,
        origin="lower",
        extent=[0, r_max, 0, z_max],
        colors="black",
        linewidths=0.5,
        linestyles="solid",
        alpha=0.8,
        zorder=1,
    )

    # contour labels
    ax.clabel(
        cs,
        inline=True,
        fontsize=8,
        fmt="%.1f",
    )

    # visitation overlay
    ax.imshow(
        blue_layer,
        origin="lower",
        extent=[0, r_max, 0, z_max],
        aspect="auto",
        zorder=2,
    )
    
    # disturbance_proxy = Line2D(
    #     [0], [0],
    #     color="black",
    #     lw=0.5,
    #     alpha=0.8,
    #     linestyle="solid",
    #     label="Disturbance contour",
    # )

    # # optional proxy for visitation
    # visit_proxy = Line2D(
    #     [0], [0],
    #     color=(0.0, 0.35, 1.0),
    #     lw=6,
    #     alpha=0.8,
    #     label="Drone visitation density",
    # )

    # ax.legend(
    #     handles=[disturbance_proxy, visit_proxy],
    #     loc="upper left",
    #     fontsize=11,
    #     frameon=True,
    #     facecolor="white",
    #     edgecolor="black",
    #     framealpha=1.0,
    # )

    # # colorbar for visit count
    # norm = Normalize(vmin=0, vmax=np.max(H))
    # sm = ScalarMappable(norm=norm, cmap="Blues")
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label("Visit count", fontsize=14)

    ax.set_xlabel("Radial Distance (m)", size=16, fontweight="bold")
    ax.set_ylabel("Altitude (m)", size=16, fontweight="bold")
    # ax.set_title(title, size=18)

    # --- 11 tick marks (10 intervals) ---
    ticks = np.linspace(0, r_max, 11)
    plt.xticks(ticks)
    plt.yticks(ticks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path

def get_episode_csvs(eval_dir):
    eval_dir = Path(eval_dir)
    csv_files = list(eval_dir.glob("*.csv"))

    rl_csv = None
    baseline_csv = None

    for f in csv_files:
        name = f.stem.lower()
        if name in {"ppo", "sac", "td3", "ddpg"}:
            rl_csv = f
        elif "centroid" in name or "baseline" in name:
            baseline_csv = f

    if rl_csv is None:
        raise FileNotFoundError(f"No RL csv found in {eval_dir}")
    if baseline_csv is None:
        raise FileNotFoundError(
            f"No baseline csv found in {eval_dir}. Found: {[f.name for f in csv_files]}"
        )

    return rl_csv, baseline_csv


def make_output_path(csv_path):
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_policy_heatmap.png")

def make_xy_output_path(csv_path):
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_policy_heatmap_xy.png")

def load_positions_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # split
    drone = df[df["entity_type"] == "large"]
    animal = df[df["entity_type"] == "animal"]

    # assume same ordering → just align row-wise
    drone = drone.reset_index(drop=True)
    animal = animal.reset_index(drop=True)

    n = min(len(drone), len(animal))

    dx = drone["x"].values[:n] - animal["x"].values[:n]
    dy = drone["y"].values[:n] - animal["y"].values[:n]

    r = np.sqrt(dx**2 + dy**2)
    z = drone["z"].values[:n]

    return r, z

def load_xy_positions_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    drone = df[df["entity_type"] == "large"].reset_index(drop=True)
    animal = df[df["entity_type"] == "animal"].reset_index(drop=True)

    n = min(len(drone), len(animal))
    if n < 2:
        raise ValueError(f"Need at least 2 matched drone/animal rows in {csv_path}")

    # match lengths
    drone = drone.iloc[:n].copy()
    animal = animal.iloc[:n].copy()

    # world-frame relative position
    dx = drone["x"].to_numpy(dtype=float) - animal["x"].to_numpy(dtype=float)
    dy = drone["y"].to_numpy(dtype=float) - animal["y"].to_numpy(dtype=float)

    # animal heading from movement
    ax = animal["x"].to_numpy(dtype=float)
    ay = animal["y"].to_numpy(dtype=float)

    hx = np.diff(ax)
    hy = np.diff(ay)

    # heading angle for each timestep
    heading = np.arctan2(hy, hx)

    heading = np.concatenate(([heading[0]], heading))

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    x_local = -sin_h * dx + cos_h * dy   # lateral
    y_local =  cos_h * dx + sin_h * dy   # forward

    return x_local, y_local

def plot_xy_heatmap(dx_vals, dy_vals, out_path, bins=80, cmap="jet", title="Unspecified"):
    cmap = truncate_colormap(cmap, 0.05, 1.0)
    lim = 100
    if lim <= 0:
        lim = 1.0

    H, _, _ = np.histogram2d(
        dx_vals,
        dy_vals,
        bins=bins,
        range=[[-lim, lim], [-lim, lim]],
    )

    vmin = np.min(H)
    vmax = np.max(H)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="equal",
        extent=[-lim, lim, -lim, lim],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # --- Add nav-style triangle at (0, 0) ---
    size = lim * 0.03  # scale relative to plot

    triangle_coords = np.array([
        [0, size],                 # tip (pointing +y)
        [-size * 0.6, -size],      # bottom left
        [-size * 0.1, -size * 0.6],# inner left notch
        [size * 0.1, -size * 0.6], # inner right notch
        [size * 0.6, -size],       # bottom right
    ])

    triangle = Polygon(
        triangle_coords,
        closed=True,
        facecolor="black",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
    )

    ax.add_patch(triangle)

    # --- Bold axis labels ---
    ax.set_xlabel("Relative x (m)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Relative y (m)", fontsize=16, fontweight="bold")

    # --- 11 tick marks (10 intervals) ---
    ticks = np.linspace(-lim, lim, 11)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_heatmap(r_vals, z_vals, out_path, bins=80, cmap="jet", title="Unspecified"):
    cmap = truncate_colormap(cmap, 0.05, 1.0)
    r_max = 100
    z_max = 100

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

    vmin = np.min(H)
    vmax = np.max(H)

    plt.figure(figsize=(7, 6))

    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, z_max],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel("Radial Distance (√(x²+y²))", size=16)
    plt.ylabel("Altitude (z)", size=16)
    # plt.title(title, size=18)

    cbar = plt.colorbar()
    cbar.set_label("Visit Count")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def load_positions_and_disturbance_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    drone = df[df["entity_type"] == "large"].reset_index(drop=True)
    animal = df[df["entity_type"] == "animal"].reset_index(drop=True)

    n = min(len(drone), len(animal))
    if n == 0:
        raise ValueError(f"No matched drone/animal rows found in {csv_path}")

    drone = drone.iloc[:n].copy()
    animal = animal.iloc[:n].copy()

    for col in ["x", "y", "z"]:
        drone[col] = pd.to_numeric(drone[col], errors="coerce")
        animal[col] = pd.to_numeric(animal[col], errors="coerce")

    dx = drone["x"].to_numpy(dtype=float) - animal["x"].to_numpy(dtype=float)
    dy = drone["y"].to_numpy(dtype=float) - animal["y"].to_numpy(dtype=float)
    z = drone["z"].to_numpy(dtype=float)

    candidate_cols = ["mean_disturbance", "p_disturbance", "disturbance"]

    disturbance = None
    chosen_col = None

    for candidate in candidate_cols:
        if candidate not in df.columns:
            continue

        vals = pd.to_numeric(drone[candidate], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).sum() > 0:
            disturbance = vals
            chosen_col = candidate
            break

    if disturbance is None:
        raise ValueError(
            f"No usable disturbance column found in {csv_path}. "
            f"Tried {candidate_cols}, but all were missing or non-finite on drone rows."
        )

    mask = (
        np.isfinite(dx) &
        np.isfinite(dy) &
        np.isfinite(z) &
        np.isfinite(disturbance)
    )

    dx = dx[mask]
    dy = dy[mask]
    z = z[mask]
    disturbance = disturbance[mask]

    if len(z) == 0:
        raise ValueError(
            f"All rows were filtered out in {csv_path}. "
            f"Chosen disturbance column: {chosen_col}"
        )

    r = np.sqrt(dx**2 + dy**2)

    return r, z, disturbance

def plot_disturbance_heatmap(r_vals, z_vals, disturbance_vals, out_path, bins=80, cmap="jet", title="Unspecified"):
    cmap = truncate_colormap(cmap, 0.05, 1.0)
    if len(r_vals) == 0 or len(z_vals) == 0 or len(disturbance_vals) == 0:
        raise ValueError("No valid disturbance samples to plot.")
    
    r_max = np.max(r_vals)
    z_max = np.max(z_vals)

    if r_max <= 0:
        r_max = 1.0
    if z_max <= 0:
        z_max = 1.0

    # sum disturbance per bin
    H_sum, _, _ = np.histogram2d(
        r_vals,
        z_vals,
        bins=bins,
        range=[[0, r_max], [0, z_max]],
        weights=disturbance_vals,
    )

    # count samples per bin
    H_count, _, _ = np.histogram2d(
        r_vals,
        z_vals,
        bins=bins,
        range=[[0, r_max], [0, z_max]],
    )

    # mean disturbance
    H_mean = np.divide(
        H_sum,
        H_count,
        out=np.full_like(H_sum, np.nan, dtype=float),
        where=H_count > 0,
    )

    # compute color range
    valid = np.isfinite(H_mean)
    if np.any(valid):
        vmin = np.nanmin(H_mean)
        vmax = np.nanmax(H_mean)
    else:
        vmin, vmax = 0.0, 1.0

    # set empty bins to lowest color
    H_plot = np.where(np.isfinite(H_mean), H_mean, vmin)

    plt.figure(figsize=(7, 6))

    plt.imshow(
        H_plot.T,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, z_max],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel("Radial Distance (√(x²+y²))", size=16)
    plt.ylabel("Altitude (z)", size=16)
    # plt.title(title, size=18)

    cbar = plt.colorbar()
    cbar.set_label("Mean Disturbance")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_disturbance_heatmap_from_csv(csv_path, bins=50, cmap="turbo", title="Unspecified"):
    csv_path = Path(csv_path)
    r_vals, z_vals, disturbance_vals = load_positions_and_disturbance_from_csv(csv_path)
    out_path = make_output_path(csv_path).with_name(make_output_path(csv_path).stem + "_disturbance.png")
    plot_disturbance_heatmap(r_vals, z_vals, disturbance_vals, out_path, bins=bins, cmap=cmap, title=title)
    return out_path

def plot_policy_heatmap_from_csv(csv_path, bins=50, cmap="turbo", title="Unspecified"):
    csv_path = Path(csv_path)
    r_vals, z_vals = load_positions_from_csv(csv_path)
    out_path = make_output_path(csv_path)
    plot_heatmap(r_vals, z_vals, out_path, bins=bins, cmap=cmap, title=title)
    return out_path

def plot_xy_policy_heatmap_from_csv(csv_path, bins=50, cmap="turbo", title="Unspecified"):
    csv_path = Path(csv_path)
    dx_vals, dy_vals = load_xy_positions_from_csv(csv_path)
    out_path = make_xy_output_path(csv_path)
    plot_xy_heatmap(dx_vals, dy_vals, out_path, bins=bins, cmap=cmap, title=title)
    return out_path

def plot_reward_heatmap_from_csv(csv_path, bins=80, cmap="YlGn", vmin=0.0, vmax=1.0, use_radial=True, title="Unspecified"):
    """
    Plot a dense reward heatmap similar in style to the visitation heatmap.

    Color in each bin = mean step reward of drone positions in that bin.
    Unvisited bins are filled with vmin so the whole plot is rendered.
    """
    cmap = truncate_colormap(cmap, 0.05, 1.0)

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
        suffix = "_reward_heatmap_rz.png"
    else:
        x_plot = x
        y_plot = y
        xlabel = "x"
        ylabel = "y"
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

    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    # plt.title(title, size=18)

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
        default=50,
        help="Number of histogram bins",
    )

    return parser.parse_args()


def main():
    args = _init_argparse()

    cmap = "Blues"

    out_path_rz = make_output_path(args.csv)
    r_vals, z_vals = load_positions_from_csv(args.csv)
    plot_heatmap(r_vals, z_vals, out_path_rz, bins=args.bins, cmap=cmap)

    out_path_xy = plot_xy_policy_heatmap_from_csv(args.csv, bins=args.bins, cmap=cmap)
    _ = plot_disturbance_heatmap_from_csv(args.csv, bins=args.bins, cmap=cmap)
    out_path_rz_d = plot_visitation_on_disturbance_background(args.csv, bins=args.bins, disturbance_cmap="bone_r")

    print(f"Saved heatmap to {out_path_rz}")
    print(f"Saved XY heatmap to {out_path_xy}")
    print(f"Saved rz_d heatmap tp {out_path_rz_d}")


if __name__ == "__main__":
    main()