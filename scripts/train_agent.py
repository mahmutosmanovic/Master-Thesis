# script folder
from scripts.config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
import collections
import numpy as np
from box import Box
from tqdm import tqdm

import os
import neptune
from neptune.utils import stringify_unsupported
from dotenv import load_dotenv
load_dotenv()


def main(config):
    NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
    API_TOKEN = os.getenv("API_TOKEN")
    run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=API_TOKEN)
    run["parameters"] = stringify_unsupported(config)

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
                        run["train/reward"].append(avg)
                        pbar.set_postfix({"Avg100": f"{avg:.2f}"})
                        
                        episode_reward = 0
                        obs, info = env.reset()
                        
                last_value = agent.get_last_value(obs, done)
                agent.learn(last_value)
            
            agent.save_models()
    finally:
        run.stop()

if __name__ == "__main__":
    main(cfg_train)
    