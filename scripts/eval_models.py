import csv
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from box import Box
from tqdm import tqdm

from environment import Env
from model import PPOAgent, MAPPOAgent, SACAgent, DQNAgent
from .run_utils import load_run, create_eval_dir, save_config_snapshot
from .centroid import CentroidStandoff
from .plots.reward_distribution import plot_eval_reward_distribution
from .plots.policy_heatmap import (
    plot_policy_heatmap_from_csv,
    plot_xy_policy_heatmap_from_csv,
    plot_reward_heatmap_from_csv,
    plot_disturbance_heatmap_from_csv,
    plot_visitation_on_disturbance_background,
    plot_closest_animal_visitation_by_drone_from_csv,
)

BASELINES = {
    "centroid": CentroidStandoff,
}

STEP_LOG_FIELDS = [
    "episode",
    "step",
    "entity_type",
    "id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "speed",
    "state",
    "reward",
    "calm_frac",
    "avoid_frac",
    "flee_frac",
    "disturbance",
    "view_x",
    "view_y",
    "view_z",
    "mean_disturbance",
    "r_monitoring",
    "p_disturbance",
    "r_vis",
    "r_dist",
    "r_align",
    "closest_animal_distance",
]

EPISODE_LOG_FIELDS = [
    "episode",
    "reward",
    "calm_frac",
    "avoid_frac",
    "flee_frac",
    "r_monitoring",
    "p_disturbance",
    "r_vis",
    "r_dist",
    "r_align",
]

SUMMARY_APPEND_FIELDS = [
    "run_name",
    "checkpoint_name",
    "train_step",
    "seed",
    "episode",
    "step",
    "reward",
    "r_monitoring",
    "p_disturbance",
    "r_dist",
]


def run_episode_summary(
    env,
    config,
    seed,
    agent=None,
    agent_type=None,
    baseline=None,
    step_writer=None,
    randomize_dr_scale=False,
):
    if randomize_dr_scale:
        env.D_R_scale = env.env_rng.uniform(0.22, 1.0)

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    ep_reward = 0.0

    while not (terminated or truncated):
        if agent is not None:
            with torch.no_grad():
                action = choose_action(agent, obs, agent_type)
        else:
            action = baseline.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if step_writer is not None:
            write_log_rows(step_writer, env.step_log(), STEP_LOG_FIELDS)

        ep_reward += float(reward)

    reward_stats = env.get_reward_stats()

    return {
        "episode": getattr(env, "episode", 1),
        "r_monitoring": float(reward_stats.get("r_monitoring", np.nan)),
        "p_disturbance": float(reward_stats.get("p_disturbance", np.nan)),
        "r_dist": float(reward_stats.get("r_dist", np.nan)),
    }


def evaluate_checkpoint_prefix_episodes(
    env,
    config,
    run_dir,
    run_name,
    checkpoint_prefix,
    seeds,
    append_summary_csv,
    randomize_dr_scale=False,
):
    weight_files = list_matching_actor_weight_files(run_dir, checkpoint_prefix)

    if not weight_files:
        raise FileNotFoundError(
            f"No actor checkpoints found in {run_dir} with prefix '{checkpoint_prefix}'. "
            f"Expected files like {checkpoint_prefix}*.pt"
        )

    print(f"Found {len(weight_files)} matching actor checkpoints:")
    for w in weight_files:
        print(f"  - {w}")

    total = len(weight_files) * len(seeds)
    f, writer = init_append_logger(append_summary_csv, SUMMARY_APPEND_FIELDS)

    try:
        with tqdm(total=total, desc="Checkpoint episode sweep") as pbar:
            for weight_file in weight_files:
                agent, agent_type, train_step = init_agent_actor_only(
                    config=config,
                    run_dir=run_dir,
                    actor_weight_file=weight_file,
                )

                env.reset_episode_id()

                for seed in seeds:
                    ep_row = run_episode_summary(
                        env=env,
                        config=config,
                        seed=seed,
                        agent=agent,
                        agent_type=agent_type,
                        randomize_dr_scale=randomize_dr_scale,
                    )

                    out_row = {
                        "run_name": run_name,
                        "checkpoint_name": weight_file,
                        "train_step": train_step,
                        "seed": seed,
                        **ep_row,
                    }
                    write_log_row(writer, out_row, SUMMARY_APPEND_FIELDS)

                    f.flush()
                    pbar.update(1)

    finally:
        f.close()

    return {
        "mode": "checkpoint_prefix_episodes",
        "run_name": run_name,
        "checkpoint_prefix": checkpoint_prefix,
        "num_checkpoints": len(weight_files),
        "num_episodes": len(seeds),
        "summary_csv": str(append_summary_csv),
    }


def init_logger(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def init_append_logger(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if (not file_exists) or path.stat().st_size == 0:
        writer.writeheader()

    return f, writer


def write_log_rows(writer, rows, fieldnames):
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_log_row(writer, row, fieldnames):
    writer.writerow({k: row.get(k, "") for k in fieldnames})


def init_agent(config, run_dir, weight_type="last"):
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
    elif agent_type == "sac":
        agent = SACAgent(config)
    elif agent_type == "dqn":
        agent = DQNAgent(config)
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
    elif agent_type == "dqn":
        agent.q_net.chkpt_dir = run_dir
        agent.target_q_net.chkpt_dir = run_dir

    agent.load_models(name=weight_type)

    if agent_type == "ppo":
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == "mappo":
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == "sac":
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        agent.target_critic_1.eval()
        agent.target_critic_2.eval()
    elif agent_type == "dqn":
        agent.q_net.eval()
        agent.target_q_net.eval()

    return agent, agent_type


def init_agent_actor_only(config, run_dir, actor_weight_file):
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
    elif agent_type == "sac":
        agent = SACAgent(config)
    elif agent_type == "dqn":
        agent = DQNAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    weight_path = Path(run_dir) / actor_weight_file
    if not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    map_location = torch.device("cpu")
    if agent_type == "ppo":
        map_location = agent.actor.device
    elif agent_type == "mappo":
        map_location = agent.actor.device
    elif agent_type == "sac":
        map_location = agent.actor.device
    elif agent_type == "dqn":
        map_location = agent.q_net.device

    checkpoint = torch.load(weight_path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint format in {weight_path}. "
            f"Expected dict checkpoint."
        )

    train_step = checkpoint.get("train_step", None)

    if agent_type in ("ppo", "mappo", "sac"):
        if "actor_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {weight_path} does not contain 'actor_state_dict'. "
                f"Keys found: {list(checkpoint.keys())}"
            )
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.actor.eval()

    elif agent_type == "dqn":
        if "q_net_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {weight_path} does not contain 'q_net_state_dict'. "
                f"Keys found: {list(checkpoint.keys())}"
            )
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.q_net.eval()

    return agent, agent_type, train_step


def choose_action(agent, obs, agent_type):
    if agent_type == "ppo":
        actions = []
        for drone_obs in obs:
            action, _, _ = agent.choose_action(drone_obs, deterministic=True)
            actions.append(action)
        return np.array(actions, dtype=np.float32)

    elif agent_type == "mappo":
        with torch.no_grad():
            actions, _, _ = agent.choose_actions(obs, deterministic=True)
        return np.asarray(actions, dtype=np.float32)

    elif agent_type == "sac":
        obs_arr = np.asarray(obs, dtype=np.float32)
        joint_obs = obs_arr.reshape(-1)

        joint_action_flat, _, _ = agent.choose_action(joint_obs, deterministic=True)
        env_action = np.asarray(joint_action_flat, dtype=np.float32).reshape(obs_arr.shape[0], -1)
        return env_action

    elif agent_type == "dqn":
        obs_arr = np.asarray(obs, dtype=np.float32)

        if obs_arr.shape[0] != 1:
            raise ValueError(
                "DQN evaluation currently expects exactly 1 drone, "
                f"but got obs shape {obs_arr.shape}"
            )

        single_obs = obs_arr[0]
        action_vec, _, _ = agent.choose_action(single_obs, deterministic=True)
        env_action = np.asarray(action_vec, dtype=np.float32).reshape(1, -1)
        return env_action

    else:
        raise ValueError(agent_type)


def run_episode(
    env,
    config,
    seed,
    agent=None,
    agent_type=None,
    baseline=None,
    step_writer=None,
    randomize_dr_scale=False,
):
    if randomize_dr_scale:
        env.D_R_scale = env.env_rng.uniform(0.22, 1.0)

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    ep_reward = 0.0

    while not (terminated or truncated):
        if agent is not None:
            with torch.no_grad():
                action = choose_action(agent, obs, agent_type)
        else:
            action = baseline.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if step_writer is not None:
            write_log_rows(step_writer, env.step_log(), STEP_LOG_FIELDS)

        ep_reward += float(reward)

    return ep_reward / config.max_episode_steps


def run_n_steps(env, seed, n_steps, agent=None, agent_type=None, baseline=None, randomize_dr_scale=False):
    if randomize_dr_scale:
        env.D_R_scale = env.env_rng.uniform(0.22, 1.0)

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    rows = []

    for step_idx in range(1, n_steps + 1):
        if terminated or truncated:
            break

        if agent is not None:
            with torch.no_grad():
                action = choose_action(agent, obs, agent_type)
        else:
            action = baseline.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        reward_stats = env.get_reward_stats()

        rows.append({
            "episode": getattr(env, "episode", 1),
            "step": step_idx,
            "reward": float(reward),
            "r_monitoring": float(reward_stats.get("r_monitoring", np.nan)),
            "p_disturbance": float(reward_stats.get("p_disturbance", np.nan)),
            "r_dist": float(reward_stats.get("r_dist", np.nan)),
        })

    return rows


def list_matching_actor_weight_files(run_dir, checkpoint_prefix):
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob(f"{checkpoint_prefix}*.pt"))
    return [p.name for p in paths if p.is_file()]


def evaluate_checkpoint_prefix_steps(
    env,
    config,
    run_dir,
    run_name,
    checkpoint_prefix,
    num_steps,
    seeds,
    append_summary_csv,
    randomize_dr_scale=False,
):
    weight_files = list_matching_actor_weight_files(run_dir, checkpoint_prefix)

    if not weight_files:
        raise FileNotFoundError(
            f"No actor checkpoints found in {run_dir} with prefix '{checkpoint_prefix}'. "
            f"Expected files like {checkpoint_prefix}*.pt"
        )

    print(f"Found {len(weight_files)} matching actor checkpoints:")
    for w in weight_files:
        print(f"  - {w}")

    total = len(weight_files) * len(seeds)
    f, writer = init_append_logger(append_summary_csv, SUMMARY_APPEND_FIELDS)

    try:
        with tqdm(total=total, desc="Checkpoint step sweep") as pbar:
            for weight_file in weight_files:
                agent, agent_type, train_step = init_agent_actor_only(
                    config=config,
                    run_dir=run_dir,
                    actor_weight_file=weight_file,
                )

                for seed in seeds:
                    step_rows = run_n_steps(
                        env=env,
                        seed=seed,
                        n_steps=num_steps,
                        agent=agent,
                        agent_type=agent_type,
                        randomize_dr_scale=randomize_dr_scale,
                    )

                    for row in step_rows:
                        out_row = {
                            "run_name": run_name,
                            "checkpoint_name": weight_file,
                            "train_step": train_step,
                            "seed": seed,
                            **row,
                        }
                        write_log_row(writer, out_row, SUMMARY_APPEND_FIELDS)

                    f.flush()
                    pbar.update(1)

    finally:
        f.close()

    return {
        "mode": "checkpoint_prefix_steps",
        "run_name": run_name,
        "checkpoint_prefix": checkpoint_prefix,
        "num_checkpoints": len(weight_files),
        "num_steps": num_steps,
        "num_seeds": len(seeds),
        "summary_csv": str(append_summary_csv),
    }


def evaluate(env, config, seeds, agent, agent_type, baseline=None, log_dir=None, randomize_dr_scale=False):
    total = len(seeds) if baseline is None else 2 * len(seeds)

    rl_rewards = []
    base_rewards = []

    with tqdm(total=total, desc="Evaluation") as pbar:
        if baseline is not None:
            env.reset_episode_id()
            f, writer = init_logger(log_dir / f"{type(baseline).__name__}.csv", STEP_LOG_FIELDS)
            ep_f, ep_writer = init_logger(log_dir / f"{type(baseline).__name__}_ep.csv", EPISODE_LOG_FIELDS)

            temp_action_type = env.config.model.space.action_type
            try:
                env.config.model.space.action_type = "rel"

                for seed in seeds:
                    base_r = run_episode(
                        env=env,
                        config=config,
                        seed=seed,
                        baseline=baseline,
                        step_writer=writer,
                        randomize_dr_scale=randomize_dr_scale,
                    )
                    base_rewards.append(base_r)

                    behavior_stats = env.get_behavior_stats()
                    reward_stats = env.get_reward_stats()

                    ep_row = {
                        "episode": env.episode,
                        "reward": base_r,
                        **behavior_stats,
                        **reward_stats,
                    }
                    write_log_row(ep_writer, ep_row, EPISODE_LOG_FIELDS)
                    pbar.update(1)
            finally:
                f.close()
                ep_f.close()
                env.config.model.space.action_type = temp_action_type

        env.reset_episode_id()
        f, writer = init_logger(log_dir / f"{agent_type}.csv", STEP_LOG_FIELDS)
        ep_f, ep_writer = init_logger(log_dir / f"{agent_type}_ep.csv", EPISODE_LOG_FIELDS)

        try:
            for seed in seeds:
                rl_r = run_episode(
                    env=env,
                    config=config,
                    seed=seed,
                    agent=agent,
                    agent_type=agent_type,
                    step_writer=writer,
                    randomize_dr_scale=randomize_dr_scale,
                )
                rl_rewards.append(rl_r)

                behavior_stats = env.get_behavior_stats()
                reward_stats = env.get_reward_stats()

                ep_row = {
                    "episode": env.episode,
                    "reward": rl_r,
                    **behavior_stats,
                    **reward_stats,
                }
                write_log_row(ep_writer, ep_row, EPISODE_LOG_FIELDS)
                pbar.update(1)
        finally:
            f.close()
            ep_f.close()

    report = {
        "rl": {
            "mean": float(np.mean(rl_rewards)),
            "std": float(np.std(rl_rewards)),
            "n": len(rl_rewards),
            "csv": str(log_dir / f"{agent_type}.csv"),
            "episode_csv": str(log_dir / f"{agent_type}_ep.csv"),
        }
    }

    if baseline is not None:
        report["baseline"] = {
            "name": type(baseline).__name__,
            "mean": float(np.mean(base_rewards)),
            "std": float(np.std(base_rewards)),
            "n": len(base_rewards),
            "csv": str(log_dir / f"{type(baseline).__name__}.csv"),
            "episode_csv": str(log_dir / f"{type(baseline).__name__}_ep.csv"),
        }

    return report


def plot_results(base_rewards, rl_rewards):
    plt.figure(figsize=(7, 5))

    rl_mean = rl_rewards.mean()
    rl_std = rl_rewards.std()

    xmin = rl_rewards.min()
    xmax = rl_rewards.max()

    if base_rewards is not None:
        xmin = min(xmin, base_rewards.min())
        xmax = max(xmax, base_rewards.max())

    x = np.linspace(xmin, xmax, 400)

    rl_pdf = (
        1 / (rl_std * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - rl_mean) / rl_std) ** 2)
    )

    plt.plot(
        x,
        rl_pdf,
        color="tab:red",
        linewidth=2,
        label=f"RL (μ={rl_mean:.3f}, σ={rl_std:.3f})",
    )
    plt.fill_between(x, rl_pdf, alpha=0.2, color="tab:red")

    if base_rewards is not None:
        base_mean = base_rewards.mean()
        base_std = base_rewards.std()

        base_pdf = (
            1 / (base_std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - base_mean) / base_std) ** 2)
        )

        plt.plot(
            x,
            base_pdf,
            color="tab:blue",
            linewidth=2,
            label=f"Baseline (μ={base_mean:.3f}, σ={base_std:.3f})",
        )
        plt.fill_between(x, base_pdf, alpha=0.2, color="tab:blue")

    plt.xlabel("Normalized Episode Reward")
    plt.ylabel("Probability Density")
    plt.title("Reward Distribution (Baseline vs RL)" if base_rewards is not None else "RL Reward Distribution")
    plt.legend()
    plt.grid(alpha=0.25)

    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/baseline.png", dpi=200, bbox_inches="tight")
    plt.close()


def _init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run folder name or 'latest'",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINES.keys()),
        default=None,
        help="Baseline policy",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="best",
        help="best or latest weights",
    )

    parser.add_argument(
        "--plot-rewards",
        action="store_true",
        help="Create reward distribution plot after evaluation",
    )

    parser.add_argument(
        "--plot-heatmaps",
        action="store_true",
        help="Create policy heatmaps from generated step logs",
    )

    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=None,
        help="Evaluate all actor checkpoint files matching this prefix, e.g. ppo_ppo_",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of inference steps per checkpoint in checkpoint sweep mode",
    )

    parser.add_argument(
        "--append-summary-csv",
        type=str,
        default=None,
        help="Single CSV file to append summary rows to in checkpoint sweep mode",
    )

    parser.add_argument(
        "--checkpoint-episodes",
        action="store_true",
        help="In checkpoint sweep mode, evaluate full episodes and append one row per episode",
    )

    parser.add_argument(
        "--randomize_dr_scale",
        action="store_true",
        help="Enable reward tradeoff randomization",
    )

    return parser.parse_args()


def _safe_make_plot(label, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        print(f"[ok] {label}: {out}")
        return out
    except Exception as e:
        print(f"[failed] {label}: {e}")
        return None


def main():
    args = _init_argparse()

    cfg, run_dir = load_run(args.run)
    config = Box(cfg)
    behavior_name = str(config._config_name)

    env = Env(config)
    env.enable_step_logging = True

    run_dir = Path(run_dir)
    run_name = run_dir.name

    if args.checkpoint_prefix is not None:
        if args.append_summary_csv is None:
            raise ValueError(
                "--append-summary-csv is required when using --checkpoint-prefix"
            )

        seeds = [args.start_seed + i for i in range(args.num_episodes)]

        print(f"Run dir: {run_dir}")
        print(f"Checkpoint prefix: {args.checkpoint_prefix}")
        print(f"Seeds: {seeds}")
        print(f"Append CSV: {args.append_summary_csv}")

        if args.checkpoint_episodes:
            print("Running checkpoint-prefix episode evaluation")

            report = evaluate_checkpoint_prefix_episodes(
                env=env,
                config=config,
                run_dir=run_dir,
                run_name=run_name,
                checkpoint_prefix=args.checkpoint_prefix,
                seeds=seeds,
                append_summary_csv=args.append_summary_csv,
                randomize_dr_scale=args.randomize_dr_scale,
            )

            print("\n=== CHECKPOINT EPISODE SWEEP DONE ===")
            print(f"Run: {report['run_name']}")
            print(f"Prefix: {report['checkpoint_prefix']}")
            print(f"Checkpoints: {report['num_checkpoints']}")
            print(f"Episodes/checkpoint: {report['num_episodes']}")
            print(f"Summary CSV: {report['summary_csv']}")
            print("EVAL_DIR::APPEND_ONLY")
            return

        print("Running checkpoint-prefix step evaluation")
        print(f"Num steps per checkpoint: {args.num_steps}")

        report = evaluate_checkpoint_prefix_steps(
            env=env,
            config=config,
            run_dir=run_dir,
            run_name=run_name,
            checkpoint_prefix=args.checkpoint_prefix,
            num_steps=args.num_steps,
            seeds=seeds,
            append_summary_csv=args.append_summary_csv,
            randomize_dr_scale=args.randomize_dr_scale,
        )

        print("\n=== CHECKPOINT STEP SWEEP DONE ===")
        print(f"Run: {report['run_name']}")
        print(f"Prefix: {report['checkpoint_prefix']}")
        print(f"Checkpoints: {report['num_checkpoints']}")
        print(f"Seeds: {report['num_seeds']}")
        print(f"Steps/checkpoint: {report['num_steps']}")
        print(f"Summary CSV: {report['summary_csv']}")
        print("EVAL_DIR::APPEND_ONLY")
        return

    agent, agent_type = init_agent(config, run_dir, args.weights)
    print(f"Detected RL algorithm: {agent_type}")

    baseline = None
    if args.baseline is not None:
        baseline_class = BASELINES[args.baseline]
        baseline = baseline_class(config)

    seeds = [args.start_seed + i for i in range(args.num_episodes)]

    print(f"Running {args.num_episodes} episodes")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}")

    eval_dir = create_eval_dir(config, args.start_seed)
    save_config_snapshot(cfg, eval_dir)

    report = evaluate(
        env=env,
        config=config,
        seeds=seeds,
        agent=agent,
        agent_type=agent_type,
        baseline=baseline,
        log_dir=eval_dir,
        randomize_dr_scale=args.randomize_dr_scale,
    )

    print("\n=== RESULTS ===")
    print(
        f">>> RL ({agent_type}): "
        f"mean={report['rl']['mean']:.4f} "
        f"std={report['rl']['std']:.4f} "
        f"n={report['rl']['n']}"
    )

    if "baseline" in report:
        print(
            f">>> Baseline ({report['baseline']['name']}): "
            f"mean={report['baseline']['mean']:.4f} "
            f"std={report['baseline']['std']:.4f} "
            f"n={report['baseline']['n']}"
        )

    if args.plot_heatmaps:
        rl_csv = eval_dir / f"{agent_type}.csv"
        print(f"\nGenerating heatmaps from: {rl_csv}")

        df = pd.read_csv(rl_csv)
        reward_min = pd.to_numeric(df["reward"], errors="coerce").min()
        reward_max = pd.to_numeric(df["reward"], errors="coerce").max()

        _safe_make_plot(
            "RL policy heatmap",
            plot_policy_heatmap_from_csv,
            rl_csv,
            bins=100,
            cmap="turbo",
            title=behavior_name,
        )
        _safe_make_plot(
            "RL disturbance heatmap",
            plot_disturbance_heatmap_from_csv,
            rl_csv,
            bins=100,
            cmap="turbo",
            title=behavior_name,
        )
        _safe_make_plot(
            "RL XY heatmap",
            plot_xy_policy_heatmap_from_csv,
            rl_csv,
            bins=100,
            cmap="turbo",
            title=behavior_name,
        )
        _safe_make_plot(
            "RL reward heatmap",
            plot_reward_heatmap_from_csv,
            rl_csv,
            bins=100,
            cmap="turbo",
            vmin=float(reward_min),
            vmax=float(reward_max),
            use_radial=True,
            title=behavior_name,
        )
        _safe_make_plot(
            "RL visitation-on-disturbance heatmap",
            plot_visitation_on_disturbance_background,
            csv_path=rl_csv,
            bins=100,
            disturbance_cmap="bone_r",
            title=behavior_name,
        )
        _safe_make_plot(
            "RL closest-animal distance heatmap",
            plot_closest_animal_visitation_by_drone_from_csv,
            rl_csv,
            title=f"",
        )

        if baseline is not None:
            baseline_csv = eval_dir / f"{type(baseline).__name__}.csv"
            print(f"\nGenerating baseline heatmaps from: {baseline_csv}")

            df = pd.read_csv(baseline_csv)
            reward_min = pd.to_numeric(df["reward"], errors="coerce").min()
            reward_max = pd.to_numeric(df["reward"], errors="coerce").max()

            _safe_make_plot(
                "Baseline policy heatmap",
                plot_policy_heatmap_from_csv,
                baseline_csv,
                bins=100,
                cmap="turbo",
                title=behavior_name,
            )
            _safe_make_plot(
                "Baseline disturbance heatmap",
                plot_disturbance_heatmap_from_csv,
                baseline_csv,
                bins=100,
                cmap="turbo",
                title=behavior_name,
            )
            _safe_make_plot(
                "Baseline XY heatmap",
                plot_xy_policy_heatmap_from_csv,
                baseline_csv,
                bins=100,
                cmap="turbo",
                title=behavior_name,
            )
            _safe_make_plot(
                "Baseline reward heatmap",
                plot_reward_heatmap_from_csv,
                baseline_csv,
                bins=100,
                cmap="turbo",
                vmin=float(reward_min),
                vmax=float(reward_max),
                use_radial=True,
                title=behavior_name,
            )
            _safe_make_plot(
                "Baseline visitation-on-disturbance heatmap",
                plot_visitation_on_disturbance_background,
                csv_path=baseline_csv,
                bins=100,
                disturbance_cmap="bone_r",
                title=behavior_name,
            )
            _safe_make_plot(
                "Baseline closest-animal distance heatmap",
                plot_closest_animal_visitation_by_drone_from_csv,
                baseline_csv,
                title=f"{behavior_name} - closest animal distance",
            )

    if args.plot_rewards and baseline is not None:
        baseline_ep_csv = eval_dir / f"{type(baseline).__name__}_ep.csv"
        rl_ep_csv = eval_dir / f"{agent_type}_ep.csv"
        _ = plot_eval_reward_distribution(rl_ep_csv, baseline_ep_csv)

    print(f"EVAL_DIR::{eval_dir.name}")


if __name__ == "__main__":
    main()