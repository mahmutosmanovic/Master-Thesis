from pathlib import Path
import math
import yaml
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("evals")
RUNS = [
    "CRW_seed42_2026-04-01_11-43-06",
    "CRW_seed42_2026-04-01_12-59-03",
    "CRW_seed42_2026-04-01_13-06-43",
    "CRW_seed42_2026-04-01_15-10-54"
]

METRICS = ["p_disturbance", "r_monitoring", "reward"]
METRICS_LUT = {"p_disturbance": "Disturbance penalty", "r_monitoring": "Monitoring reward", "reward": "Episode reward"}

def ci95(series):
    series = pd.Series(series).dropna()
    n = len(series)
    if n <= 1:
        return 0.0
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


rows = []

for run in RUNS:
    run_dir = ROOT / run

    # find csv
    csv_path = None
    for name in ["mappo_ep.csv", "ppo_ep.csv"]:
        p = run_dir / name
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        print(f"Skipping {run}: no _ep csv found")
        continue

    # load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Skipping {run}: no config.yaml found")
        continue

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # x-axis: use animal count as proxy for number of drones
    x_val = cfg["animal"]["env"]["count"]

    df = pd.read_csv(csv_path)

    row = {
        "run": run,
        "x": x_val,
        "n": len(df),
    }

    for metric in METRICS:
        row[f"{metric}_mean"] = df[metric].mean()
        row[f"{metric}_ci"] = ci95(df[metric])

    rows.append(row)

plot_df = pd.DataFrame(rows).sort_values("x")

print(plot_df.to_string(index=False))

plt.figure(figsize=(10, 5))

for metric in METRICS:
    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci"

    plt.plot(plot_df["x"], plot_df[mean_col], marker="o", label=METRICS_LUT[metric])
    plt.fill_between(
        plot_df["x"],
        plot_df[mean_col] - plot_df[ci_col],
        plot_df[mean_col] + plot_df[ci_col],
        alpha=0.2,
    )

plt.xlabel("Number of drones")
plt.ylabel("Metric value")
plt.title("Mean ± 95% CI across 30 samples")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/mappo.png", dpi=300)