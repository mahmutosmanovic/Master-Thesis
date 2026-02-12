# script folder
from scripts.config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
import numpy as np
from tqdm import trange, tqdm

def main(config):
    env = Env(cfg_train, render_mode=None)
    obs, info = env.reset()
    
    for traj in range(config["agent"]["ep"]):
        for step in range(config["steps"]):
            actions = env.sample_action()
            observation, reward, terminated, truncated, info = env.step(actions)


if __name__ == "__main__":
    main(cfg_train)
    