import argparse
import os
import torch

from box import Box

from environment import Env
from model import PPOAgent, MAPPOAgent
from .run_utils import load_run


def init_agent(config, run_dir, name):
    """
    Load agent based on the stored run configuration.
    """

    config.run_dir = run_dir
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.load_models(name=name)

    agent.actor.eval()
    agent.critic.eval()

    return agent, agent_type


def choose_action(agent, obs, agent_type):
    """
    Handles PPO vs MAPPO action inference.
    """

    if agent_type == "ppo":

        actions = []

        for drone_obs in obs:
            action, _, _ = agent.choose_action(drone_obs, deterministic=True)
            actions.append(action)

        return actions

    elif agent_type == "mappo":

        action, _, _ = agent.choose_action(obs, deterministic=True)
        return action

    else:
        raise ValueError(agent_type)


def main(config, run_dir, seed, model_type="best"):

    config = Box(config)

    env = Env(config, render_mode="human", seed=seed)
    obs, info = env.reset()

    agent, agent_type = init_agent(config, run_dir, model_type)

    print(f"Loaded agent type: {agent_type}")

    terminated = False
    truncated = False

    step_count = 0
    episode_reward = 0.0

    while not (terminated or truncated):

        with torch.no_grad():
            action = choose_action(agent, obs, agent_type)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        episode_reward += reward

    norm_reward = episode_reward / config.max_episode_steps

    print(f"Episode finished in {step_count} steps")
    print(f"Normalized Reward: {norm_reward:.4f}")

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
        help="Random seed for environment",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="best",
        help="best or latest weights",
    )


    return parser.parse_args()


if __name__ == "__main__":

    args = _init_argparse()

    cfg, run_dir = load_run(args.run)

    main(cfg, run_dir, seed=args.seed, model_type=args.weights)