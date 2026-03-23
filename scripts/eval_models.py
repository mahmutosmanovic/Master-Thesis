import csv
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from box import Box
from tqdm import tqdm

from environment import Env
from model import PPOAgent, MAPPOAgent, ConstrainedPPOAgent
from .run_utils import load_run, create_eval_dir, save_config_snapshot
from .centroid import CentroidStandoff
from .plots.reward_distribution import plot_eval_reward_distribution
from .plots.policy_heatmap import (
    plot_policy_heatmap_from_csv,
    plot_xy_policy_heatmap_from_csv,
    plot_reward_heatmap_from_csv,
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
    "cost",
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
    "r_bucket",
]

EPISODE_LOG_FIELDS = [
    "episode",
    "reward",
    "cost",
    "calm_frac",
    "avoid_frac",
    "flee_frac",
    "r_monitoring",
    "p_disturbance",
    "r_vis",
    "r_dist",
    "r_align",
    "r_bucket",
]

def init_logger(path, fieldnames):
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def write_log_rows(writer, rows, fieldnames):
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_log_row(writer, row, fieldnames):
    writer.writerow({k: row.get(k, "") for k in fieldnames})


def init_agent(config, run_dir, weight_type="last", device="cpu"):
    """
    Initialize agent from run config and load weights.
    Assumes PPO / MAPPO / constrained PPO implement load_models(name=...).
    """
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config, device=device)

    elif agent_type == "mappo":
        agent = MAPPOAgent(config, device=device)

    elif agent_type == "constrained_ppo":
        agent = ConstrainedPPOAgent(config, device=device)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # force checkpoint directory to selected run
    if hasattr(agent, "actor"):
        agent.actor.chkpt_dir = run_dir

    if hasattr(agent, "critic"):
        agent.critic.chkpt_dir = run_dir

    if hasattr(agent, "reward_critic"):
        agent.reward_critic.chkpt_dir = run_dir

    if hasattr(agent, "cost_critic"):
        agent.cost_critic.chkpt_dir = run_dir

    agent.load_models(name=weight_type)

    if hasattr(agent, "actor"):
        agent.actor.eval()

    if hasattr(agent, "critic"):
        agent.critic.eval()

    if hasattr(agent, "reward_critic"):
        agent.reward_critic.eval()

    if hasattr(agent, "cost_critic"):
        agent.cost_critic.eval()

    return agent, agent_type


def choose_action(agent, obs, agent_type):
    """
    Handles PPO / constrained PPO / MAPPO inference logic.
    """
    if agent_type in {"ppo", "constrained_ppo"}:
        actions = []

        for drone_obs in obs:
            out = agent.choose_action(drone_obs, deterministic=True)
            action = out[0]
            actions.append(action)

        return np.array(actions, dtype=np.float32)

    elif agent_type == "mappo":
        with torch.no_grad():
            actions, _, _ = agent.choose_action(obs, deterministic=True)
        return actions

    else:
        raise ValueError(agent_type)


def run_episode(env, config, seed, agent=None, agent_type=None, baseline=None, step_writer=None):
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    ep_reward = 0.0
    ep_cost = 0.0

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
        ep_cost += float(info.get("cost", 0.0))

    return {
        "reward": ep_reward / config.max_episode_steps,
        "cost": ep_cost / config.max_episode_steps,
    }


def evaluate(env, config, seeds, agent, agent_type, baseline=None, log_dir=None):
    total = len(seeds) if baseline is None else 2 * len(seeds)

    rl_rewards = []
    rl_costs = []

    base_rewards = []
    base_costs = []

    with tqdm(total=total, desc="Evaluation") as pbar:
        if baseline is not None:
            env.reset_episode_id()
            f, writer = init_logger(log_dir / f"{type(baseline).__name__}.csv", STEP_LOG_FIELDS)
            ep_f, ep_writer = init_logger(log_dir / f"{type(baseline).__name__}_ep.csv", EPISODE_LOG_FIELDS)

            temp_action_type = env.config.model.space.action_type
            try:
                env.config.model.space.action_type = "abs"

                for seed in seeds:
                    base_out = run_episode(
                        env,
                        config,
                        seed,
                        baseline=baseline,
                        step_writer=writer,
                    )
                    base_rewards.append(base_out["reward"])
                    base_costs.append(base_out["cost"])

                    behavior_stats = env.get_behavior_stats() or {}
                    reward_stats = env.get_reward_stats() or {}

                    ep_row = {
                        "episode": env.episode,
                        "reward": base_out["reward"],
                        "cost": base_out["cost"],
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
                rl_out = run_episode(
                    env,
                    config,
                    seed,
                    agent=agent,
                    agent_type=agent_type,
                    step_writer=writer,
                )
                rl_rewards.append(rl_out["reward"])
                rl_costs.append(rl_out["cost"])

                behavior_stats = env.get_behavior_stats() or {}
                reward_stats = env.get_reward_stats() or {}

                ep_row = {
                    "episode": env.episode,
                    "reward": rl_out["reward"],
                    "cost": rl_out["cost"],
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
            "cost_mean": float(np.mean(rl_costs)),
            "cost_std": float(np.std(rl_costs)),
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
            "cost_mean": float(np.mean(base_costs)),
            "cost_std": float(np.std(base_costs)),
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
    plt.title("Reward Distribution" if base_rewards is None else "Reward Distribution (Baseline vs RL)")
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
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on",
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

    return parser.parse_args()


def main():
    args = _init_argparse()

    cfg, run_dir = load_run(args.run)
    config = Box(cfg)

    env = Env(config)
    env.enable_step_logging = True

    agent, agent_type = init_agent(config, run_dir, args.weights, device=args.device)

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
        env,
        config,
        seeds,
        agent,
        agent_type,
        baseline,
        log_dir=eval_dir,
    )

    print("\n=== RESULTS ===")
    print(
        f">>> RL ({agent_type}): "
        f"mean={report['rl']['mean']:.4f} "
        f"std={report['rl']['std']:.4f} "
        f"cost_mean={report['rl']['cost_mean']:.4f} "
        f"cost_std={report['rl']['cost_std']:.4f} "
        f"n={report['rl']['n']}"
    )

    if "baseline" in report:
        print(
            f">>> Baseline ({report['baseline']['name']}): "
            f"mean={report['baseline']['mean']:.4f} "
            f"std={report['baseline']['std']:.4f} "
            f"cost_mean={report['baseline']['cost_mean']:.4f} "
            f"cost_std={report['baseline']['cost_std']:.4f} "
            f"n={report['baseline']['n']}"
        )

    if args.plot_rewards and baseline is not None:
        _ = plot_eval_reward_distribution(eval_dir)

    if args.plot_heatmaps:
        rl_csv = eval_dir / f"{agent_type}.csv"

        _ = plot_policy_heatmap_from_csv(rl_csv)
        _ = plot_xy_policy_heatmap_from_csv(rl_csv)
        _ = plot_reward_heatmap_from_csv(
            rl_csv,
            bins=60,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            use_radial=True,
        )

        if baseline is not None:
            baseline_csv = eval_dir / f"{type(baseline).__name__}.csv"

            _ = plot_policy_heatmap_from_csv(baseline_csv)
            _ = plot_xy_policy_heatmap_from_csv(baseline_csv)
            _ = plot_reward_heatmap_from_csv(
                baseline_csv,
                bins=60,
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
                use_radial=True,
            )

    print(f"EVAL_DIR::{eval_dir.name}")


if __name__ == "__main__":
    main()