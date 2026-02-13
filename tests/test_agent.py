# script folder
from scripts.config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
from box import Box
import numpy as np
from tqdm import trange, tqdm

def main(config):
    config = Box(config)

    env = Env(cfg_train, render_mode=None)
    agent = Agent(config)

    obs, info = env.reset()
    agent.add_to_buffer(obs)

    total_steps = 0

    while total_steps < config.model.sampling.total_timesteps:

        for _ in range(config.model.sampling.rollout_steps):

            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(next_obs, reward, terminated)

            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                obs, info = env.reset()
                agent.add_to_buffer(obs)

        # --- UPDATE PPO ---
        agent.learn()
        agent.clear_buffer()


if __name__ == "__main__":
    main(cfg_train)
    