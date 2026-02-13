import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment.resource import ResourceField
from experiments.run_behaviours import envconf

def main():
    df = pd.read_csv("logs/simulations/behaviour_test.csv")
    seed_df = pd.read_csv("logs/simulations/behaviour_test_seeds.csv")
    episode = 10

    row = seed_df.loc[seed_df["episode"] == episode]
    row = row.iloc[0]

    episode_seed  = int(row["seed"])
    resource_seed = int(row["resource_seed"])
    
    res = ResourceField(envconf.map_width, envconf.map_height, envconf.resource_frequency, envconf.resource_scale, envconf.resource_abundance, seed=resource_seed)

    summary = calculate_statistics(df)
    print("\nFinal summary:")
    print(summary)
    plot_episode(df[df["episode"] == episode], res)


def calculate_statistics(df):
    encounters_per_ep = (
        df.groupby(["episode", "behaviour"])["encounter"]
        .sum()
        .reset_index()
    )

    encounter_stats = (
        encounters_per_ep
        .groupby("behaviour")["encounter"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    encounter_stats = encounter_stats.rename(columns={
        "mean": "mean_encounters",
        "std": "std_encounters",
        "count": "n_episodes"
    })

    total_by_state = (
        df.groupby(["behaviour", "behaviour_state"])
        .size()
        .reset_index(name="total_timesteps")
    )

    total_by_behaviour = (
        df.groupby("behaviour")
        .size()
        .reset_index(name="total_steps")
    )

    state_fraction = total_by_state.merge(
        total_by_behaviour,
        on="behaviour"
    )

    state_fraction["fraction"] = (
        state_fraction["total_timesteps"] /
        state_fraction["total_steps"]
    )

    state_pivot = state_fraction.pivot(
        index="behaviour",
        columns="behaviour_state",
        values="fraction"
    ).reset_index()

    # Rename columns nicely
    state_pivot.columns.name = None

    state_pivot = state_pivot.rename(
        columns=lambda c: f"percent_{c}" if c != "behaviour" else c
    )
    state_pivot = state_pivot.fillna(0)

    summary = encounter_stats.merge(
        state_pivot,
        on="behaviour",
        how="left"
    )

    return summary

def plot_episode(df, res):
    xs = np.linspace(0, envconf.map_width, 200)
    ys = np.linspace(0, envconf.map_width, 200)

    Z = np.zeros((200, 200))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[i, j] = res.p_resource((x, y))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z.T,
        origin="lower",
        extent=[0, envconf.map_width, 0, envconf.map_width],
        interpolation="bilinear",
        cmap="gray",
        alpha=0.9
    )
    plt.colorbar(label="p(encounter)")

    groups = df.groupby("agent_id", sort=False)

    for agent_id, d in groups:
        x = d["x"].to_numpy(float)
        y = d["y"].to_numpy(float)

        label = f"agent {agent_id}"
        if "behaviour" in d.columns:
            b = d["behaviour"].dropna()
            if len(b) > 0:
                label = f"{b.iloc[0]} (id {agent_id})"

        plt.plot(x, y, linewidth=2, label=label)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title("Episode trajectories (XY) with resource field")
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs("figs", exist_ok=True)
    save_path = os.path.join("figs", "example_behaviour.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()