# config
import argparse
from config import load_config

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

    env = Env(config, render_mode="human", seed=99)
    obs, info = env.reset()

    agent = Agent(config)

    model_dir = "tmp/ppo"

    actor_path = os.path.join(model_dir, "actor_torch_ppo.pt")
    agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.actor.eval()

    terminated = False
    truncated = False

    step_count = 0
    episode_reward = 0.0

    while not (terminated or truncated):

        with torch.no_grad():
            action, _, _ = agent.choose_action(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        episode_reward += reward

    norm_reward = episode_reward / config.max_episode_steps
    print(f"Episode finished. Total Reward: {norm_reward:.4f}")

    env.viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="train",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Env. config: {cfg['_config_name']}")
    main(cfg)