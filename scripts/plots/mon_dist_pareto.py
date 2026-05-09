import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_manifest(manifest_name):
    manifest_path = Path("pareto") / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return pd.read_csv(manifest_path).dropna()


def load_eval_stats(eval_dir):
    csv_path = eval_dir / "ppo_ep.csv"

    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    return {
        "monitor": df["r_monitoring"],
        "disturbance": df["p_disturbance"],
    }


def main(manifest_name):

    manifest = load_manifest(manifest_name)

    monitors = []
    disturbances = []
    labels = []

    for _, row in manifest.iterrows():

        eval_dir = Path("evals") / row["eval_name"]
        config = row["config"]

        stats = load_eval_stats(eval_dir)

        if stats is None:
            continue

        monitors.append(stats["monitor"])
        disturbances.append(stats["disturbance"])

    plt.figure(figsize=(6,6))

    plt.scatter(monitors, disturbances)

    for x, y, label in zip(monitors, disturbances, labels):
        plt.text(x, y, label)

    plt.xlabel("Monitoring reward")
    plt.ylabel("Disturbance")
    plt.title("Monitoring vs Disturbance")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Manifest filename inside pareto/")
    args = parser.parse_args()

    main(args.manifest)