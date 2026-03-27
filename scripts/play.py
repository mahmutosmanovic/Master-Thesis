import argparse
import os
import torch

from box import Box
import numpy as np
from environment import Env
from model import PPOAgent, MAPPOAgent, SACAgent
from .run_utils import load_run


def init_agent(config, run_dir, weight_type="last"):
    """
    Standard full-agent loading using the project's normal load_models(name=...).
    """
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
    elif agent_type == "sac":
        agent = SACAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if agent_type == "ppo":
        agent.actor.chkpt_dir = run_dir
        agent.critic.chkpt_dir = run_dir
    elif agent_type == "mappo":
        agent.actor.chkpt_dir = run_dir
        agent.critic.chkpt_dir = run_dir
    elif agent_type == "sac":
        agent.actor.chkpt_dir = run_dir
        agent.critic_1.chkpt_dir = run_dir
        agent.critic_2.chkpt_dir = run_dir
        agent.target_critic_1.chkpt_dir = run_dir
        agent.target_critic_2.chkpt_dir = run_dir

    agent.load_models(name=weight_type)

    agent.actor.eval()

    if agent_type == "ppo":
        agent.critic.eval()
    elif agent_type == "mappo":
        agent.critic.eval()
    elif agent_type == "sac":
        agent.critic_1.eval()
        agent.critic_2.eval()
        agent.target_critic_1.eval()
        agent.target_critic_2.eval()

    return agent, agent_type


def choose_action(agent, obs, agent_type):
    if agent_type == "ppo":
        actions = []
        for drone_obs in obs:
            action, _, _ = agent.choose_action(drone_obs, deterministic=True)
            actions.append(action)
        return np.array(actions, dtype=np.float32)

    elif agent_type == "mappo":
        with torch.no_grad():
            actions, _, _ = agent.choose_action(obs, deterministic=True)
        return np.asarray(actions, dtype=np.float32)

    elif agent_type == "sac":
        obs_arr = np.asarray(obs, dtype=np.float32)
        joint_obs = obs_arr.reshape(-1)

        joint_action_flat, _, _ = agent.choose_action(joint_obs, deterministic=True)
        
        env_action = np.asarray(joint_action_flat, dtype=np.float32).reshape(obs_arr.shape[0], -1)
        return env_action

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