# experiments/run_ppo.py
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from environment.utils.utils import decode_action
from environment.agents.behaviour import BehaviourConfig, CRWConfig, ExploreExploitConfig, TraplineConfig
from environment import Environment, EnvConfig, DroneConfig, AnimalConfig
from experiments.settings import rand5jack1drone, standard_animal
from environment.resource_map import ResourceMap

envconf = EnvConfig(
    # simulation
    dt=0.2,
    max_t=1000.0,

    # map spawning bounds
    map_size=1000.0,
    map_altitude=100.0,

    # animals
    force_bounds=True,
    animals=[
        dict(config=standard_animal(), count=1, behaviour_cfg=CRWConfig()),
        dict(config=standard_animal(), count=1, behaviour_cfg=ExploreExploitConfig()),
        dict(config=standard_animal(), count=1, behaviour_cfg=TraplineConfig()),
        # dict(params=jackal_params(), count=1, behaviour_cfg=LEARN()),
    ],

    p_wavelenght = 200.0,
    p_reduction = 0.2,
    p_scale = 0.4,
    sample_res = 5.0,
    min_poi_p = 1e-2,
    kernel_size = 250.0,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--n_eps", type=int, default=1)
    parser.add_argument("--log", type=str, default="logs/simulations/behaviour_test.csv")
    args = parser.parse_args()

    logs = {}
    seeds = {}

    env = Environment(envconf)
    _, info = env.reset(seed=args.seed)

    for ep in tqdm(range(args.n_eps)):
        done = False
        while not done:
            _, _, done, _ = env.step({})

            if done:
                logs[ep] = env.log.copy()
                seeds[ep] = {"seed": info["seed"], "resource_seed": info["resource_seed"]}
                _, info = env.reset()
                continue
    
    os.makedirs("logs/simulations", exist_ok=True)
    
    dfs = []
    for ep, log in logs.items():
        df_ep = pd.DataFrame(log)
        df_ep["episode"] = ep
        dfs.append(df_ep)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("logs/simulations/behaviour_test.csv")

    seed_df = (
        pd.DataFrame.from_dict(seeds, orient="index")
        .reset_index()
        .rename(columns={"index": "episode"})
    )

    seed_df.to_csv("logs/simulations/behaviour_test_seeds.csv", index=False)

    plot_and_stats()

def plot_and_stats():
    df = pd.read_csv("logs/simulations/behaviour_test.csv")
    seed_df = pd.read_csv("logs/simulations/behaviour_test_seeds.csv")
    episode = 0

    row = seed_df.loc[seed_df["episode"] == episode]
    row = row.iloc[0]

    episode_seed  = int(row["seed"])
    resource_seed = int(row["resource_seed"])
    
    res = ResourceMap(
            p_wavelenght=envconf.p_wavelenght,
            p_reduction=envconf.p_reduction,
            p_scale=envconf.p_scale,
            sample_res=envconf.sample_res,
            min_poi_p=envconf.min_poi_p,
            world_size=envconf.map_size,
            kernel_size=envconf.kernel_size,
            seed=resource_seed
            )

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
    p_map, x_space, y_space = res.sample_map()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(
        p_map.T,
        origin="lower",
        extent=[min(x_space), max(x_space), min(y_space), max(y_space)],
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

    plt.xlim(0, res.world_size)
    plt.ylim(0, res.world_size)

    plt.title("Episode trajectories (XY) with resource field")
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs("figs", exist_ok=True)
    save_path = os.path.join("figs", "example_behaviour.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()