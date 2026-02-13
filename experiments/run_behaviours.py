# experiments/run_ppo.py
import os
import argparse
import numpy as np
import torch
import pandas as pd

from tqdm import tqdm

from utils.utils import decode_action
from environment.agents.behaviour import BehaviourConfig, CRWConfig, ExploreExploitConfig, TraplineConfig
from environment import Environment, EnvConfig, DroneParams, AnimalParams
from experiments.settings import rand5jack1drone, jackal_params

envconf = EnvConfig(
    # simulation
    dt=0.2,
    max_t=1000.0,

    # map spawning bounds
    map_width=1000.0,
    map_height=1000.0,
    map_altitude=100.0,

    # animals
    animals=[
        dict(params=jackal_params(), count=1, behaviour_cfg=CRWConfig()),
        dict(params=jackal_params(), count=1, behaviour_cfg=ExploreExploitConfig()),
        dict(params=jackal_params(), count=1, behaviour_cfg=TraplineConfig()),
        # dict(params=jackal_params(), count=1, behaviour_cfg=LEARN()),
    ],

    resource_frequency = 0.006,
    resource_scale = 0.35,
    resource_abundance = 0.4,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--n_eps", type=int, default=100)
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


if __name__ == "__main__":
    main()