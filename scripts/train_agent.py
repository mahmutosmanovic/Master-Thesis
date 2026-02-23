# script folder
from scripts.config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
import subprocess
import collections
from box import Box
from tqdm import tqdm
from pathlib import Path

import numpy as np
import time
import os
import neptune
from neptune.utils import stringify_unsupported
from dotenv import load_dotenv
load_dotenv()

def init_neptune_log(run, config):
        # parameters upload
        run["parameters"] = stringify_unsupported(config)

        # source code upload
        project_root = Path(__file__).resolve().parents[1]
        code_dirs = [
            project_root / "scripts",
            project_root / "environment",
            project_root / "model",
        ]
        files_to_upload = []
        for d in code_dirs:
            files_to_upload.extend(str(p) for p in d.rglob("*.py"))
        run["source_code/files"].upload_files(files_to_upload)

        commit = subprocess.getoutput("git rev-parse HEAD")
        run["source_code/git_commit"] = commit
    

def main(config, neptune_logging=False):
    if neptune_logging:
        NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
        API_TOKEN = os.getenv("API_TOKEN")
        run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=API_TOKEN)
        init_neptune_log(run, config)

    config = Box(config)
    env = Env(config)
    agent = Agent(config)
    obs, info = env.reset()

    total_steps = 0
    episode_reward = 0
    reward_all_100 = []
    reward_queue = collections.deque(maxlen=100)
    try:
        with tqdm(total=config.model.sampling.total_timesteps, desc="Training") as pbar:
            while total_steps < config.model.sampling.total_timesteps:

                for _ in range(config.model.sampling.rollout_steps):
                    total_steps += 1
                    pbar.update(1)   

                    action, log_prob, val = agent.choose_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    done = terminated or truncated
                    agent.remember(obs, action, log_prob, val, reward, done)
                    
                    obs = next_obs

                    episode_reward += reward
                    if terminated or truncated:
                        reward_queue.append(episode_reward / config.max_episode_steps)

                        avg = np.mean(reward_queue)
                        reward_all_100.append(avg)
                        stats = env.get_behavior_stats()
                        if neptune_logging: 
                            run["train_stats/reward"].append(avg)

                            run["train_stats/calm_frac"].append(stats["calm_frac"])
                            run["train_stats/avoid_frac"].append(stats["avoid_frac"])
                            run["train_stats/flee_frac"].append(stats["flee_frac"])
                            run["train_stats/mean_disturbance"].append(stats["mean_disturbance"])

                        pbar.set_postfix({
                            "rew_100": f"{avg:.2f}",
                            "mean_dist": f"{stats['mean_disturbance']:.2f}",
                        })
                        episode_reward = 0
                        obs, info = env.reset()
                        
                last_value = agent.get_last_value(obs, done)
                agent.learn(last_value)
            
            agent.save_models()
    finally:
        if neptune_logging:
            run.stop()

if __name__ == "__main__":
    main(cfg_train, neptune_logging=True)
    