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
import csv
import argparse
import subprocess
import numpy as np
import wandb as wb
from box import Box
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).resolve().parents[1]


def init_wandb(config, agent_type):
    api_token = os.getenv("API_TOKEN")
    wb.login(key=api_token)

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    now = datetime.now()
    timestamp = now.strftime("%B%d_t%H%M").lower()
    run_name = f"train_{agent_type}_seed{config['seed']}_{timestamp}"

    run = wb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=config,
        tags=[agent_type, f"seed{config['seed']}"],
    )

    source_files = []
    for folder in ["config", "environment", "model", "scripts"]:
        root = project_root / folder
        source_files += list(root.rglob("*.py"))
        source_files += list(root.rglob("*.yaml"))
        source_files += list(root.rglob("*.yml"))

    artifact = wb.Artifact("source_code", type="code")

    for f in sorted(source_files):
        rel_path = f.relative_to(project_root)
        artifact.add_file(str(f), name=str(rel_path))

    wb.log_artifact(artifact)

    commit = subprocess.getoutput("git rev-parse HEAD")
    wb.config.update({"git_commit": commit}, allow_val_change=True)

    return run


def _init_agent(config, agent_type, device):
    if agent_type == "ppo":
        return PPOAgent(config, device)
    elif agent_type == "mappo":
        return MAPPOAgent(config, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _get_behavior_name(config):
    behavior_raw = str(config.animal.init.behavior)
    return behavior_raw.split("_CFG")[0]


def _get_episode_csv_path(config):
    behavior_name = _get_behavior_name(config)
    return Path(config.run_dir) / f"{behavior_name}_episode.csv"


def _get_train_csv_path(config):
    behavior_name = _get_behavior_name(config)
    return Path(config.run_dir) / f"{behavior_name}_train.csv"


# =========================
# LOCAL CSV: EPISODE LOGGING
# =========================
def _init_episode_csv(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "episode_reward_norm",
            "episode_progress",
            "calm_frac",
            "avoid_frac",
            "flee_frac",
            "r_monitoring",
            "p_disturbance",
            "r_vis",
            "r_dist",
            "r_align",
            "r_bucket",
            "checkpoint_save_count",
        ])


def _append_episode_csv(csv_path, step, episode_reward_norm, stats, r_stats, save_count):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            episode_reward_norm,
            r_stats["episode_progress"],
            stats["calm_frac"],
            stats["avoid_frac"],
            stats["flee_frac"],
            r_stats["r_monitoring"],
            r_stats["p_disturbance"],
            r_stats["r_vis"],
            r_stats["r_dist"],
            r_stats["r_align"],
            r_stats["r_bucket"],
            save_count,
        ])


# =======================
# LOCAL CSV: TRAIN LOGGING
# =======================
def _init_train_csv(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "train_entropy_coef",
            "train_policy_entropy",
            "actor_loss",
            "critic_loss",
            "actor_lr",
            "critic_lr",
        ])


def _append_train_csv(csv_path, step, train_metrics):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            train_metrics["train_entropy_coef"],
            train_metrics["train_policy_entropy"],
            train_metrics["actor_loss"],
            train_metrics["critic_loss"],
            train_metrics["actor_lr"],
            train_metrics["critic_lr"],
        ])


# ======================
# WRAPPERS: EPISODE LOGS
# ======================
def _log_episode_local(csv_path, step, episode_reward_norm, stats, r_stats, save_count):
    _append_episode_csv(
        csv_path=csv_path,
        step=step,
        episode_reward_norm=episode_reward_norm,
        stats=stats,
        r_stats=r_stats,
        save_count=save_count,
    )


def _log_episode_wandb(step, episode_reward_norm, stats, r_stats, save_count):
    wb.log({
        "episode/reward_norm": episode_reward_norm,
        "episode/progress": r_stats["episode_progress"],
        "episode/step": step,

        "behavior/calm_frac": stats["calm_frac"],
        "behavior/avoid_frac": stats["avoid_frac"],
        "behavior/flee_frac": stats["flee_frac"],

        "reward/monitoring": r_stats["r_monitoring"],
        "reward/disturbance_penalty": r_stats["p_disturbance"],
        "reward/visibility": r_stats["r_vis"],
        "reward/distance": r_stats["r_dist"],
        "reward/alignment": r_stats["r_align"],
        "reward/bucket": r_stats["r_bucket"],

        "checkpoint/save_count": save_count,
    }, step=step)


# ====================
# WRAPPERS: TRAIN LOGS
# ====================
def _log_train_local(csv_path, step, train_metrics):
    _append_train_csv(csv_path=csv_path, step=step, train_metrics=train_metrics)


def _log_train_wandb(step, train_metrics):
    wb.log({
        "train/entropy_coef": train_metrics["train_entropy_coef"],
        "train/policy_entropy": train_metrics["train_policy_entropy"],
        "train/actor_loss": train_metrics["actor_loss"],
        "train/critic_loss": train_metrics["critic_loss"],
        "train/actor_lr": train_metrics["actor_lr"],
        "train/critic_lr": train_metrics["critic_lr"],
    }, step=step)


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


def main(config, agent_type="ppo", use_wandb=False, device="cpu"):
    if use_wandb:
        _ = init_wandb(config, agent_type)

    config = Box(config)

    env = Env(config)
    agent = _init_agent(config, agent_type, device)

    episode_csv_path = _get_episode_csv_path(config)
    train_csv_path = _get_train_csv_path(config)

    _init_episode_csv(episode_csv_path)
    _init_train_csv(train_csv_path)

    obs, info = env.reset()
    done = False

    curr_steps = 0
    episode_reward = 0.0
    max_rew = -np.inf

    save_count = 0
    save_models_frac = 0.5
    total_steps = config.model.sampling.total_timesteps

    try:
        with tqdm(total=total_steps, desc="Training") as pbar:
            while curr_steps < total_steps:
                rollout_steps = min(
                    config.model.sampling.rollout_steps,
                    total_steps - curr_steps
                )

                for _ in range(rollout_steps):
                    curr_steps += 1
                    pbar.update(1)

                    next_obs, reward, done, terminated, truncated, info = agent_env_step(
                        agent, env, obs, agent_type
                    )

                    obs = next_obs
                    episode_reward += reward

                    if terminated or truncated:
                        episode_reward_norm = episode_reward / config.max_episode_steps
                        stats = env.get_behavior_stats()
                        r_stats = env.get_reward_stats()

                        _log_episode_local(episode_csv_path, curr_steps, episode_reward_norm, stats, r_stats, save_count)
                        if use_wandb:
                            _log_episode_wandb(curr_steps, episode_reward_norm, stats, r_stats, save_count)

                        pbar.set_postfix({
                            "rew_100": f"{episode_reward_norm:.2f}",
                            "mean_dist": f"{r_stats['p_disturbance']:.2f}",
                        })

                        if (
                            episode_reward_norm > max_rew
                            and curr_steps >= save_models_frac * total_steps
                        ):
                            agent.save_models(name="best")
                            max_rew = episode_reward_norm
                            save_count += 1

                        episode_reward = 0.0
                        obs, info = env.reset()
                        done = False

                last_value = agent.get_last_value(obs, done)
                train_metrics = agent.learn(last_value)

                if train_metrics is not None:
                    _log_train_local(train_csv_path, curr_steps, train_metrics)
                    if use_wandb:
                        _log_train_wandb(curr_steps, train_metrics)

            agent.save_models(name="last")

            if use_wandb:
                wb.save(os.path.join(config.run_dir, "*"))

    finally:
        if use_wandb:
            wb.finish()


def _init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train", help="Config name inside config/ folder")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "mappo"], help="RL agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _init_argparse()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    cfg["agent_type"] = args.agent

    run_dir = create_run_dir(cfg, args.seed)

    save_config_snapshot(cfg, run_dir)
    cfg["run_dir"] = str(run_dir)

    main(cfg, agent_type=args.agent, use_wandb=args.wandb, device=args.device)
    print(f"RUN_DIR::{run_dir.name}")