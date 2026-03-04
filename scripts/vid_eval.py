# config
import argparse
from .run_utils import load_run

# environment
from environment import Env

# model
from model import Agent

# standard
from box import Box
import torch
import os


def main(config, model_dir, seed):

    config = Box(config)

    env = Env(config, render_mode="human", seed=seed)
    obs, info = env.reset()

    agent = Agent(config)

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


def _init_argparse():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run folder name or 'latest'",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed for environment (default: 99)",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _init_argparse()

    cfg, run_dir = load_run(args.run)

    main(cfg, run_dir, seed=args.seed)