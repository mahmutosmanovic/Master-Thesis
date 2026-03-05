# config
from config import load_config
from .run_utils import create_run_dir, save_config_snapshot

# environment
from environment import Env

# model
from model import PPOAgent
from model import MAPPOAgent

# standard modules
import os
import time
import neptune
import argparse
import subprocess
import collections
import numpy as np
from box import Box
from tqdm import tqdm
from pathlib import Path
from neptune.utils import stringify_unsupported
from dotenv import load_dotenv
load_dotenv()

def init_neptune_log(run, config, agent_type):
        # parameters upload
        config["agent_type"] = agent_type
        run["parameters"] = stringify_unsupported(config)

        # source code upload
        project_root = Path(__file__).resolve().parents[1]
        code_dirs = [
            project_root / "scripts",
            project_root / "environment",
            project_root / "model",
            project_root / "config",
        ]
        files_to_upload = []
        for d in code_dirs:
            files_to_upload.extend(str(p) for p in d.rglob("*.py"))
            files_to_upload.extend(str(p) for p in d.rglob("*.yaml"))
            files_to_upload.extend(str(p) for p in d.rglob("*.yml"))
        run["source_code/files"].upload_files(files_to_upload)

        commit = subprocess.getoutput("git rev-parse HEAD")
        run["source_code/git_commit"] = commit
    
def _init_agent(config, agent_type):
    if agent_type == "ppo":
        return PPOAgent(config)
    elif agent_type == "mappo":
        return MAPPOAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
def agent_env_step(agent, env, obs, agent_type):
    """
    Executes one environment interaction step for PPO or MAPPO.

    Returns:
        next_obs, reward, done, terminated, truncated, info
    """

    if agent_type == "ppo":

        actions = []
        logps = []
        vals = []

        for drone_obs in obs:
            a, lp, v = agent.choose_action(drone_obs)
            actions.append(a)
            logps.append(lp)
            vals.append(v)

        actions = np.array(actions)

        next_obs, reward, terminated, truncated, info = env.step(actions)

        done = terminated or truncated

        for i in range(len(obs)):
            agent.remember(obs[i], actions[i], logps[i], vals[i], reward, done)

    elif agent_type == "mappo":

        actions, log_prob, val = agent.choose_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(actions)

        done = terminated or truncated

        agent.remember(obs, actions, log_prob, val, reward, done)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return next_obs, reward, done, terminated, truncated, info

def main(config, agent_type="ppo", neptune_logging=False):
    if neptune_logging:
        NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
        API_TOKEN = os.getenv("API_TOKEN")
        run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=API_TOKEN)
        init_neptune_log(run, config, agent_type)

    config = Box(config)
    env = Env(config)
    agent = _init_agent(config, agent_type)
    obs, info = env.reset()
    done = False

    total_steps = 0
    episode_reward = 0
    reward_all_100 = []
    reward_queue = collections.deque(maxlen=100)
    try:
        with tqdm(total=config.model.sampling.total_timesteps, desc="Training") as pbar:
            while total_steps < config.model.sampling.total_timesteps:

                for _ in range(config.model.sampling.rollout_steps):
                    total_steps += 1
                    pbar.update(1)   

                    next_obs, reward, done, terminated, truncated, info = agent_env_step(
                        agent, env, obs, agent_type
                    )
                    
                    obs = next_obs
                    episode_reward += reward

                    if terminated or truncated:
                        reward_queue.append(episode_reward / config.max_episode_steps)

                        avg = np.mean(reward_queue)
                        reward_all_100.append(avg)
                        stats = env.get_behavior_stats()
                        if neptune_logging: 
                            run["train_stats/reward"].append(avg)

                            run["train_stats/calm_frac"].append(stats["calm_frac"])
                            run["train_stats/avoid_frac"].append(stats["avoid_frac"])
                            run["train_stats/flee_frac"].append(stats["flee_frac"])
                            run["train_stats/mean_disturbance"].append(stats["mean_disturbance"])

                        pbar.set_postfix({
                            "rew_100": f"{avg:.2f}",
                            "mean_dist": f"{stats['mean_disturbance']:.2f}",
                        })
                        episode_reward = 0
                        obs, info = env.reset()
                        done = False
                        
                last_value = agent.get_last_value(obs, done)
                agent.learn(last_value)
            
            agent.save_models()
    finally:
        if neptune_logging:
            run.stop()

def _init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="Config name inside config/ folder",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "mappo"],
        help="RL agent type (default: ppo)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--neptune",
        action="store_true",
        help="Enable Neptune logging",
    )

    return parser.parse_args()
    
if __name__ == "__main__":

    args = _init_argparse()

    cfg = load_config(args.config)

    run_dir = create_run_dir(cfg, args.seed)
    save_config_snapshot(cfg, run_dir)

    cfg["run_dir"] = str(run_dir)
    cfg["seed"] = args.seed
    cfg["agent_type"] = args.agent

    main(cfg, agent_type=args.agent, neptune_logging=args.neptune)