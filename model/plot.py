import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        ...

    def reward(self, xs, ys, path="./tmp/ppo/reward.png"):
        df = pd.DataFrame({
            "step": xs,
            "reward": ys
        })

        plt.figure(figsize=(10, 6))
        plt.plot(
            df["step"],
            df["reward"],
            linewidth=2,
            color="#cd00d4",
        )
        plt.xlabel("Step", fontsize=12, fontweight="bold")
        plt.ylabel("Reward", fontsize=12, fontweight="bold")
        plt.title("Training Rewards", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()