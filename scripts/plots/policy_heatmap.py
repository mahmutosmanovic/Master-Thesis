import csv
import argparse
from pathlib import Path

import numpy as np
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