# script folder
from .config import c1
from .immutables import Behavior, MovementDim

# environment folder
from environment.env import Env

def main(config):
    env = Env(config, seed=42)
    observation = env.reset()
    animals, drones = observation

    

if __name__ == "__main__":
    # RUN WITH:
    # python -m scripts.train
    main(c1)
