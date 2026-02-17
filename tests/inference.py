# config
from scripts.config import cfg_train

# environment
from environment import Env

# model
from model import Agent

# standard
from box import Box
import torch
import os


def main(config):

    config = Box(config)

    env = Env(config, render_mode=None)
    obs, info = env.reset()

    agent = Agent(config)

    model_dir = "tmp/ppo"

    actor_path = os.path.join(model_dir, "actor_torch_ppo.pt")
    critic_path = os.path.join(model_dir, "critic_torch_ppo.pt")

    agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))

    agent.actor.eval()
    agent.critic.eval()

    print("Models loaded successfully.")

    env.set_render_mode("human")

    terminated = False
    truncated = False
    total_reward = 0

    print("Recording episode...")

    cnt = 0

    while not (terminated or truncated):

        # Deterministic action for inference
        with torch.no_grad():
            action, _, _ = agent.choose_action(obs, deterministic=True)
            print(action)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        cnt += 1
        if cnt == 3:
            break


    print(f"Episode finished. Total Reward: {total_reward:.4f}")


if __name__ == "__main__":
    main(cfg_train)
