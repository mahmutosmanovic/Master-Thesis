# config
from config import load_config
from .run_utils import create_run_dir, save_config_snapshot

# environment
from environment import Env

# model
from model import PPOAgent, MAPPOAgent, SACAgent

# standard modules
import os
import csv
import argparse
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb as wb
from box import Box
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

project_root = Path(__file__).resolve().parents[1]


def init_wandb(config, agent_type):
    wb.login(key=os.getenv("API_TOKEN"))

    run = wb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name=f"train_{agent_type}_seed{config['seed']}_{datetime.now().strftime('%B%d_t%H%M').lower()}",
        config=config,
        tags=[agent_type, f"seed{config['seed']}"],
    )

    artifact = wb.Artifact("source_code", type="code")
    for folder in ("config", "environment", "model", "scripts"):
        for ext in ("*.py", "*.yaml", "*.yml"):
            for f in sorted((project_root / folder).rglob(ext)):
                artifact.add_file(str(f), name=str(f.relative_to(project_root)))
    wb.log_artifact(artifact)

    wb.config.update(
        {"git_commit": subprocess.getoutput("git rev-parse HEAD")},
        allow_val_change=True,
    )
    return run


def _init_agent(config, agent_type, device):
    agents = {
        "ppo": PPOAgent,
        "mappo": MAPPOAgent,
        "sac": SACAgent,
    }
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agents[agent_type](config, device)


def _behavior_name(config):
    return str(config.animal.init.behavior).split("_CFG")[0]


def _csv_paths(config):
    stem = _behavior_name(config)
    run_dir = Path(config.run_dir)
    return run_dir / f"{stem}_episode.csv", run_dir / f"{stem}_train.csv"


def append_csv_row(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a+", newline="") as f:
        f.seek(0, 2)
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def _log_episode_local(csv_path, step, episode_reward_norm, stats, r_stats, save_count):
    append_csv_row(csv_path, {
        "step": step,
        "episode_reward_norm": episode_reward_norm,
        **stats,
        **r_stats,
        "checkpoint_save_count": save_count,
    })


def _log_episode_wandb(step, episode_reward_norm, stats, r_stats, save_count, spawn_radius):
    wb.log({
        "episode/reward_norm": episode_reward_norm,
        "episode/progress": r_stats["episode_progress"],
        "episode/step": step,
        "episode/spawn_radius": spawn_radius,
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


def _log_train_local(csv_path, step, train_metrics):
    append_csv_row(csv_path, {"step": step, **train_metrics})


def _log_train_wandb(step, train_metrics):
    payload = {f"train/{k}": v for k, v in train_metrics.items() if v is not None}
    if payload:
        wb.log(payload, step=step)


def agent_env_step(agent, env, obs, agent_type):
    if agent_type == "ppo":
        actions, logps, vals = [], [], []
        for drone_obs in obs:
            a, lp, v = agent.choose_action(drone_obs)
            actions.append(a)
            logps.append(lp)
            vals.append(v)

        actions = np.asarray(actions, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        for i in range(len(obs)):
            agent.remember(obs[i], actions[i], logps[i], vals[i], reward, done)

    elif agent_type == "mappo":
        actions, log_prob, val = agent.choose_actions(obs)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        agent.remember(obs, actions, log_prob, val, reward, done)

    elif agent_type == "sac":
        joint_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        joint_action, _, _ = agent.choose_action(joint_obs, deterministic=False)
        env_action = joint_action.reshape(obs.shape[0], -1).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        joint_next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        agent.remember(joint_obs, joint_action, reward, joint_next_obs, done)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return next_obs, reward, done, terminated, truncated, info


def _should_train(agent_type, step_in_rollout, rollout_steps):
    return agent_type == "sac" or step_in_rollout == rollout_steps - 1


def _train_step(agent, agent_type, obs, done):
    if agent_type in ("ppo", "mappo"):
        return agent.learn(agent.get_last_value(obs, done))
    if agent_type == "sac":
        return agent.learn()
    raise ValueError(f"Unknown agent type: {agent_type}")


def _has_train_metrics(train_metrics):
    return train_metrics is not None and any(v is not None for v in train_metrics.values())


class SpawnRadiusSchedule:
    def __init__(self, cooldown, min_radius, step_size):
        self.cooldown = cooldown
        self.min_radius = min_radius
        self.step_size = step_size
        self.counter = 0

    def ready(self):
        if self.counter <= 0:
            return True
        self.counter -= 1
        return False

    def step(self, curr):
        self.counter = self.cooldown
        return max(self.min_radius, curr - self.step_size)


def main(config, agent_type="ppo", use_wandb=False, device="cpu"):
    config = Box(config)

    if use_wandb:
        init_wandb(config, agent_type)

    srs = SpawnRadiusSchedule(50, 200, 100) if args.schedule_spawn else None
    rewards = deque(maxlen=20)

    env = Env(config)
    agent = _init_agent(config, agent_type, device)
    episode_csv_path, train_csv_path = _csv_paths(config)

    obs, _ = env.reset()
    done = False

    curr_steps = 0
    episode_reward = 0.0
    max_rew = -np.inf
    save_count = 0
    save_models_frac = 0.5
    total_steps = config.model.sampling.total_timesteps

    milestone_thresholds = []
    milestone_saved = {thr: False for thr in milestone_thresholds}

    save_steps = []
    saved_steps = {}
    if args.save_every is not None:
        save_steps = list(np.arange(args.save_every, total_steps + 1, args.save_every))
        saved_steps = {step: False for step in save_steps}

    try:
        with tqdm(total=total_steps, desc="Training") as pbar:
            while curr_steps < total_steps:
                rollout_steps = 1 if agent_type == "sac" else min(
                    config.model.sampling.rollout_steps,
                    total_steps - curr_steps,
                )

                for step_in_rollout in range(rollout_steps):
                    curr_steps += 1
                    pbar.update(1)

                    next_obs, reward, done, terminated, truncated, _ = agent_env_step(
                        agent, env, obs, agent_type
                    )
                    obs = next_obs
                    episode_reward += reward

                    if terminated or truncated:
                        episode_reward_norm = episode_reward / config.max_episode_steps
                        stats = env.get_behavior_stats()
                        r_stats = env.get_reward_stats()

                        _log_episode_local(
                            episode_csv_path, curr_steps, episode_reward_norm, stats, r_stats, save_count
                        )
                        if use_wandb:
                            _log_episode_wandb(
                                curr_steps, episode_reward_norm, stats, r_stats, save_count, env.spawn_radius
                            )

                        pbar.set_postfix({
                            "rew_100": f"{episode_reward_norm:.2f}",
                            "mean_dist": f"{r_stats['p_disturbance']:.2f}",
                        })
                        rewards.append(episode_reward_norm)

                        if milestone_thresholds:
                            thr = milestone_thresholds[0]
                            if not milestone_saved[thr] and np.mean(rewards) >= thr:
                                tag = str(thr).replace(".", "p")
                                agent.save_models(name=f"reward_{tag}")
                                milestone_saved[thr] = True
                                save_count += 1
                                milestone_thresholds.pop(0)
                                print(f"[INFO] Saved milestone checkpoint at reward >= {thr:.1f}")

                        if save_steps:
                            step = save_steps[0]
                            if not saved_steps[step] and curr_steps >= step:
                                agent.save_models(name=f"step_{step}")
                                saved_steps[step] = True
                                save_count += 1
                                save_steps.pop(0)
                                print(f"[INFO] Saved checkpoint at step >= {step}")

                        if episode_reward_norm > max_rew and curr_steps >= save_models_frac * total_steps:
                            agent.save_models(name="best")
                            max_rew = episode_reward_norm
                            save_count += 1

                        if srs and srs.ready() and np.mean(rewards) >= 1.0:
                            env.spawn_radius = srs.step(env.spawn_radius)

                        if args.randomize_dr_scale:
                            env.alpha = env.env_rng.uniform(0.22, 1.0)

                        episode_reward = 0.0
                        obs, _ = env.reset()
                        done = False

                    if _should_train(agent_type, step_in_rollout, rollout_steps):
                        train_metrics = _train_step(agent, agent_type, obs, done)
                        if _has_train_metrics(train_metrics):
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
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "mappo", "sac"], help="RL agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--schedule_spawn", action="store_true", help="Enable animal spawn radius scheduling")
    parser.add_argument("--randomize_dr_scale", action="store_true", help="Enable reward tradeoff randomization")
    parser.add_argument("--save_every", type=int, default=None, help="Checkpoint interval in steps")
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