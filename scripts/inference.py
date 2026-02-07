# script folder
from .config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
import numpy as np
from tqdm import trange, tqdm

def main(config):
    env = Env(config, seed=42)
    agent = Agent()
    agent.load(config["model_path"])
    agent.eval()

    rewards = []
    global_step = 0
    try:
        for ep in trange(config["episodes"]):

            obs, info = env.reset() # obs: (animals, drones)
            episodic_reward = 0

            for t in range(config["steps"]):

                global_step += 1

                action = agent.policy(obs, explore=False)

                obs, reward, terminated, truncated, info = env.step(action)

                episodic_reward += reward
                
                if terminated or truncated:
                    break

            rewards.append(episodic_reward)
            tqdm.write(f"Episode {ep}: Reward = {episodic_reward:.2f}")

        tqdm.write(f"Mean reward: {np.mean(rewards):.2f}")
        tqdm.write(f"Std reward: {np.std(rewards):.2f}")

    finally:
        env.close()

if __name__ == "__main__":
    # RUN WITH:
    # python -m scripts.train
    main(cfg_train)
