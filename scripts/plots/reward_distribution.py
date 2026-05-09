import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_rewards(csv_path):
    rewards = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
    return np.array(rewards, dtype=np.float32)

def get_episode_csvs(eval_dir):
    eval_dir = Path(eval_dir)

    if (eval_dir / "ppo_ep.csv").exists():
        rl_csv = eval_dir / "ppo_ep.csv"
    else:
        rl_csv = eval_dir / "mappo_ep.csv"

    baseline_csv = next(
        path for path in eval_dir.glob("*_ep.csv")
        if path.name not in {"ppo_ep.csv", "mappo_ep.csv"}
    )

    return rl_csv, baseline_csv

def plot_results(base_rewards, rl_rewards, out_path, rl_label="RL", baseline_label="Baseline"):
    plt.figure(figsize=(7, 5))

    rl_mean = rl_rewards.mean()
    rl_std = rl_rewards.std()

    xmin = rl_rewards.min()
    xmax = rl_rewards.max()

    xmin = min(xmin, base_rewards.min())
    xmax = max(xmax, base_rewards.max())

    x = np.linspace(xmin, xmax, 400)

    rl_pdf = (
        1 / (rl_std * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - rl_mean) / rl_std) ** 2)
    )

    plt.plot(
        x,
        rl_pdf,
        color="tab:red",
        linewidth=2,
        label=f"{rl_label} (μ={rl_mean:.3f}, σ={rl_std:.3f})",
    )
    plt.fill_between(x, rl_pdf, alpha=0.2, color="tab:red")

    base_mean = base_rewards.mean()
    base_std = base_rewards.std()

    base_pdf = (
        1 / (base_std * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - base_mean) / base_std) ** 2)
    )

    plt.plot(
        x,
        base_pdf,
        color="tab:blue",
        linewidth=2,
        label=f"{baseline_label} (μ={base_mean:.3f}, σ={base_std:.3f})",
    )
    plt.fill_between(x, base_pdf, alpha=0.2, color="tab:blue")

    plt.xlabel("Normalized Episode Reward")
    plt.ylabel("Probability Density")
    plt.title(f"Reward Distribution ({baseline_label} vs {rl_label})")
    plt.legend()
    plt.grid(alpha=0.25)

    out_path = Path(out_path)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_eval_reward_distribution(rl_csv, baseline_csv):
    eval_dir = Path(rl_csv).parent

    rl_rewards = load_rewards(rl_csv)
    base_rewards = load_rewards(baseline_csv)

    out_path = eval_dir / "reward_distribution.png"

    plot_results(
        base_rewards,
        rl_rewards,
        out_path,
        rl_label=rl_csv.stem.replace("_ep", ""),
        baseline_label=baseline_csv.stem.replace("_ep", ""),
    )

    return out_path

def _init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = _init_argparse()

    eval_dir = Path(args.eval_dir)
    rl_csv, baseline_csv = get_episode_csvs(eval_dir)

    rl_rewards = load_rewards(rl_csv)
    base_rewards = load_rewards(baseline_csv)

    out_path = eval_dir / "reward_distribution.png"

    plot_results(
        base_rewards,
        rl_rewards,
        out_path,
        rl_label=rl_csv.stem.replace("_ep", ""),
        baseline_label=baseline_csv.stem.replace("_ep", ""),
    )

    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()