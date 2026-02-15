# script folder
from scripts.config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
import numpy as np
from box import Box
from tqdm import trange, tqdm

def main(config):
    config = Box(config)
    env = Env(cfg_train, render_mode=None)
    agent = Agent(config)
    obs, info = env.reset()
    agent.initialize_networks(obs)

    total_steps = 0
    while total_steps < config.model.sampling.total_timesteps:

        for _ in range(config.model.sampling.rollout_steps):
            total_steps += 1

            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.add_to_buffer(obs, action, reward, done, log_prob, value)

            obs = next_obs

            if terminated or truncated:
                obs, info = env.reset()

        # agent.learn()
        # agent.clear_buffer()

if __name__ == "__main__":
    main(cfg_train)
    