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

    while not (terminated or truncated):

        # Deterministic action for inference
        with torch.no_grad():
            # action, _, _ = agent.choose_action(obs, deterministic=True)
            action, _, _ = agent.choose_action(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
    norm_reward = total_reward / config.max_episode_steps
    print(f"Episode finished. Total Reward: {norm_reward:.4f}")

    env.viewer.close()


if __name__ == "__main__":
    main(cfg_train)
