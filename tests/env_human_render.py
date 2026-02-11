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
    env = Env(cfg_train, render_mode="human")
    obs, info = env.reset()
    
    for t in range(1, config["steps"]+1):
        if t==5:
            env.set_render_mode(None)

        if t==8:
            env.set_render_mode("human")

        actions = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(actions)

if __name__ == "__main__":
    main(cfg_train)
    