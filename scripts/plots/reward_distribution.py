import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_rewards(csv_path):
    rewards = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
    return np.asarray(rewards, dtype=np.float32)


def find_existing_csv(eval_dir, candidates):
    eval_dir = Path(eval_dir)
    for name in candidates:
        path = eval_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No matching csv found in {eval_dir}. Tried: {candidates}"
    )


def plot_eval_reward_distribution(eval_dir):
    eval_dir = Path(eval_dir)

    rl_csv = find_existing_csv(
        eval_dir,
        [
            "constrained_ppo_ep.csv",
            "ppo_ep.csv",
            "mappo_ep.csv",
        ],
    )

    baseline_csv = find_existing_csv(
        eval_dir,
        [
            "CentroidStandoff_ep.csv",
        ],
    )

    rl_rewards = load_rewards(rl_csv)
    base_rewards = load_rewards(baseline_csv)

    plt.figure(figsize=(7, 5))

    rl_mean = rl_rewards.mean()
    rl_std = rl_rewards.std()

    xmin = min(rl_rewards.min(), base_rewards.min())
    xmax = max(rl_rewards.max(), base_rewards.max())
    x = np.linspace(xmin, xmax, 400)

    if rl_std > 1e-12:
        rl_pdf = (
            1 / (rl_std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - rl_mean) / rl_std) ** 2)
        )
        plt.plot(
            x,
            rl_pdf,
            linewidth=2,
            label=f"{rl_csv.stem.replace('_ep', '')} (μ={rl_mean:.3f}, σ={rl_std:.3f})",
        )
        plt.fill_between(x, rl_pdf, alpha=0.2)
    else:
        plt.axvline(
            rl_mean,
            linewidth=2,
            label=f"{rl_csv.stem.replace('_ep', '')} (μ={rl_mean:.3f}, σ≈0)",
        )

    base_mean = base_rewards.mean()
    base_std = base_rewards.std()

    if base_std > 1e-12:
        base_pdf = (
            1 / (base_std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - base_mean) / base_std) ** 2)
        )
        plt.plot(
            x,
            base_pdf,
            linewidth=2,
            label=f"{baseline_csv.stem.replace('_ep', '')} (μ={base_mean:.3f}, σ={base_std:.3f})",
        )
        plt.fill_between(x, base_pdf, alpha=0.2)
    else:
        plt.axvline(
            base_mean,
            linewidth=2,
            label=f"{baseline_csv.stem.replace('_ep', '')} (μ={base_mean:.3f}, σ≈0)",
        )

    plt.xlabel("Normalized Episode Reward")
    plt.ylabel("Probability Density")
    plt.title("Reward Distribution")
    plt.legend()
    plt.grid(alpha=0.25)

    out_path = eval_dir / "reward_distribution.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path