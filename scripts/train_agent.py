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
import wandb
import argparse
import subprocess
import numpy as np
from box import Box
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parents[1]

def init_wandb(config, agent_type):

    api_token = os.getenv("API_TOKEN")
    wandb.login(key=api_token)

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    now = datetime.now()
    timestamp = now.strftime("%B%d_t%H%M").lower()
    run_name = f"train_{agent_type}_seed{config['seed']}_{timestamp}"

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=config,
        tags=[agent_type, f"seed{config['seed']}"],
    )

    # collect source files
    source_files = []
    for folder in ["config", "environment", "model", "scripts"]:
        root = project_root / folder
        source_files += list(root.rglob("*.py"))
        source_files += list(root.rglob("*.yaml"))
        source_files += list(root.rglob("*.yml"))

    # upload as artifact with correct paths
    artifact = wandb.Artifact("source_code", type="code")

    for f in sorted(source_files):
        rel_path = f.relative_to(project_root)
        artifact.add_file(str(f), name=str(rel_path))

    wandb.log_artifact(artifact)

    # log git commit
    commit = subprocess.getoutput("git rev-parse HEAD")
    wandb.config.update({"git_commit": commit}, allow_val_change=True)

    return run


def _init_agent(config, agent_type, device):
    if agent_type == "ppo":
        return PPOAgent(config, device)
    elif agent_type == "mappo":
        return MAPPOAgent(config,device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def agent_env_step(agent, env, obs, agent_type):

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


def main(config, agent_type="ppo", logging=False, device='cpu'):

    if logging:
        _ = init_wandb(config, agent_type)

    config = Box(config)

    env = Env(config)
    agent = _init_agent(config, agent_type, device)

    obs, info = env.reset()
    done = False

    curr_steps = 0
    episode_reward = 0
    max_rew = -np.inf

    save_count = 0
    save_models_frac = 0.5
    total_steps = config.model.sampling.total_timesteps

    try:
        with tqdm(total=total_steps, desc="Training") as pbar:

            while curr_steps < total_steps:

                for _ in range(config.model.sampling.rollout_steps):

                    curr_steps += 1
                    pbar.update(1)

                    next_obs, reward, done, terminated, truncated, info = agent_env_step(
                        agent, env, obs, agent_type
                    )

                    obs = next_obs
                    episode_reward += reward
                    episode_reward_norm = -np.inf

                    if terminated or truncated:

                        episode_reward_norm = episode_reward / config.max_episode_steps
                        stats = env.get_behavior_stats()
                        r_stats = env.get_reward_stats()

                        if logging:
                            wandb.log({
                                # episode-level summary
                                "episode/reward_norm": episode_reward_norm,
                                "episode/progress": r_stats["episode_progress"],

                                # behavior / environment response
                                "behavior/calm_frac": stats["calm_frac"],
                                "behavior/avoid_frac": stats["avoid_frac"],
                                "behavior/flee_frac": stats["flee_frac"],

                                # reward decomposition
                                "reward/monitoring": r_stats["r_monitoring"],
                                "reward/disturbance_penalty": r_stats["p_disturbance"],
                                "reward/visibility": r_stats["r_vis"],
                                "reward/distance": r_stats["r_dist"],
                                "reward/alignment": r_stats["r_align"],
                                "reward/bucket": r_stats["r_bucket"],

                                # bookkeeping
                                "checkpoint/save_count": save_count,
                                "system/step": curr_steps,
                            })

                        pbar.set_postfix({
                            "rew_100": f"{episode_reward_norm:.2f}",
                            "mean_dist": f"{r_stats['p_disturbance']:.2f}",
                        })

                        if episode_reward_norm > max_rew and curr_steps >= save_models_frac * total_steps:
                            agent.save_models(name="best")
                            max_rew = episode_reward_norm
                            save_count += 1

                        episode_reward = 0
                        obs, info = env.reset()
                        done = False

                last_value = agent.get_last_value(obs, done)
                train_metrics = agent.learn(last_value)

                env.disturb_scale = env.env_rng.uniform(low=0.1, high=1.1)
                if curr_steps % (config.model.sampling.rollout_steps * 10) == 0:
                    agent.save_models(name=f"{curr_steps}k")

                if logging and train_metrics is not None:
                    wandb.log({
                        # training / optimization
                        "train/entropy_coef": train_metrics["train_entropy_coef"],
                        "train/policy_entropy": train_metrics["train_policy_entropy"],
                        "train/actor_loss": train_metrics["actor_loss"],
                        "train/critic_loss": train_metrics["critic_loss"],
                        "train/actor_lr": train_metrics["actor_lr"],
                        "train/critic_lr": train_metrics["critic_lr"],
                    })

            agent.save_models(name="last")

            if logging:
                wandb.save(os.path.join(config.run_dir, "*"))

    finally:
        if logging:
            wandb.finish()


def _init_argparse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="train", help="Config name inside config/ folder")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "mappo"], help="RL agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging",)

    return parser.parse_args()


if __name__ == "__main__":

    args = _init_argparse()

    cfg = load_config(args.config)

    cfg["seed"] = args.seed
    cfg["agent_type"] = args.agent

    run_dir = create_run_dir(cfg, args.seed)

    save_config_snapshot(cfg, run_dir)
    cfg["run_dir"] = str(run_dir)
    main(cfg, agent_type=args.agent, logging=args.wandb, device=args.device)
    print(f"RUN_DIR::{run_dir.name}")