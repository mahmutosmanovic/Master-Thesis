import csv
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from box import Box
from tqdm import tqdm

from environment import Env
from model import PPOAgent, MAPPOAgent
from .run_utils import load_run, create_eval_dir, save_config_snapshot
from .centroid import CentroidStandoff
from .plots.reward_distribution import plot_eval_reward_distribution
from .plots.policy_heatmap import plot_policy_heatmap_from_csv


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
    "disturbance",
    "view_x",
    "view_y",
    "view_z",
]

EPISODE_LOG_FIELDS = [
    "episode",
    "reward",
    "calm_frac",
    "avoid_frac",
    "flee_frac",
    "mean_disturbance",
    "r_monitoring",
    "p_disturbance",
    "r_vis",
    "r_dist",
    "r_align",
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

def init_agent(config, run_dir):
    """
    Initialize agent from run config and load weights.
    """

    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
        actor_path = os.path.join(run_dir, "actor_torch_ppo.pt")

    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
        actor_path = os.path.join(run_dir, "actor_torch_mappo.pt")

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    state_dict = torch.load(actor_path, map_location="cpu")
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    return agent, agent_type


def choose_action(agent, obs, agent_type):
    """
    Handles PPO vs MAPPO inference logic.
    """

    if agent_type == "ppo":

        actions = []

        for drone_obs in obs:
            action, _, _ = agent.choose_action(drone_obs, deterministic=True)
            actions.append(action)

        return np.array(actions)

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


def evaluate(env, config, seeds, agent, agent_type, baseline=None, log_dir=None):
    total = len(seeds) if baseline is None else 2 * len(seeds)

    rl_rewards = []
    base_rewards = []

    with tqdm(total=total, desc="Evaluation") as pbar:

        if baseline is not None:
            env.reset_episode_id()
            f, writer = init_logger(log_dir / f"{type(baseline).__name__}.csv", STEP_LOG_FIELDS)
            ep_f, ep_writer = init_logger(log_dir / f"{type(baseline).__name__}_ep.csv", EPISODE_LOG_FIELDS)

            try:
                for seed in seeds:
                    base_r = run_episode(
                        env,
                        config,
                        seed,
                        baseline=baseline,
                        step_writer=writer,
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

        env.reset_episode_id()
        f, writer = init_logger(log_dir / f"{agent_type}.csv", STEP_LOG_FIELDS)
        ep_f, ep_writer = init_logger(log_dir / f"{agent_type}_ep.csv", EPISODE_LOG_FIELDS)

        try:
            for seed in seeds:
                rl_r = run_episode(
                    env,
                    config,
                    seed,
                    agent=agent,
                    agent_type=agent_type,
                    step_writer=writer,
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

    if base_rewards is None:
        plt.title("RL Reward Distribution")
    else:
        plt.title("Reward Distribution (Baseline vs RL)")

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

    agent, agent_type = init_agent(config, run_dir)

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
        log_dir = eval_dir
    )

    print("\n=== RESULTS ===")
    print(f"Evaluation directory: {eval_dir}")

    print(
        f">>> RL ({agent_type}): "
        f"mean={report['rl']['mean']:.4f} "
        f"std={report['rl']['std']:.4f} "
        f"n={report['rl']['n']}"
    )
    print(f"    step log:    {report['rl']['csv']}")
    print(f"    episode log: {report['rl']['episode_csv']}")

    if "baseline" in report:
        print(
            f">>> Baseline ({report['baseline']['name']}): "
            f"mean={report['baseline']['mean']:.4f} "
            f"std={report['baseline']['std']:.4f} "
            f"n={report['baseline']['n']}"
        )
        print(f"    step log:    {report['baseline']['csv']}")
        print(f"    episode log: {report['baseline']['episode_csv']}")
    
    if args.plot_rewards:
        if baseline is not None:
            reward_plot_path = plot_eval_reward_distribution(eval_dir)
            print(f"Saved reward plot to {reward_plot_path}")

    if args.plot_heatmaps:
        rl_heatmap_path = plot_policy_heatmap_from_csv(eval_dir / f"{agent_type}.csv")
        print(f"Saved RL heatmap to {rl_heatmap_path}")

        if baseline is not None:
            baseline_csv = eval_dir / f"{type(baseline).__name__}.csv"
            baseline_heatmap_path = plot_policy_heatmap_from_csv(baseline_csv)
            print(f"Saved baseline heatmap to {baseline_heatmap_path}")


if __name__ == "__main__":
    main()