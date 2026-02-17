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
    env = Env(Box(cfg_train), render_mode=None)
    obs, info = env.reset()
    print(obs)
    # env.set_render_mode(None)
    
    for step in range(1, 2):
        actions = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(actions)
        

if __name__ == "__main__":
    main(cfg_train)
    