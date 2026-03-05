import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from box import Box
from tqdm import tqdm

from environment import Env
from model import PPOAgent, MAPPOAgent
from .run_utils import load_run
from .centroid import CentroidStandoff


BASELINES = {
    "centroid": CentroidStandoff,
}


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


def run_episode(env, config, seed, agent=None, agent_type=None, baseline=None):

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

        ep_reward += float(reward)

    return ep_reward / config.max_episode_steps


def evaluate(env, config, seeds, agent, agent_type, baseline=None):

    rl_rewards = []
    base_rewards = []

    total = len(seeds) if baseline is None else 2 * len(seeds)

    with tqdm(total=total, desc="Evaluation") as pbar:

        for seed in seeds:

            if baseline is not None:
                base_r = run_episode(
                    env,
                    config,
                    seed,
                    baseline=baseline
                )
                base_rewards.append(base_r)
                pbar.update(1)

            rl_r = run_episode(
                env,
                config,
                seed,
                agent=agent,
                agent_type=agent_type,
            )

            rl_rewards.append(rl_r)
            pbar.update(1)

    if baseline is None:
        return None, np.array(rl_rewards)

    return np.array(base_rewards), np.array(rl_rewards)


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
        "--save_plot",
        action="store_true",
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

    base_rewards, rl_rewards = evaluate(
        env,
        config,
        seeds,
        agent,
        agent_type,
        baseline,
    )

    print("\n=== RESULTS ===")

    print(
        f">>> RL: mean={rl_rewards.mean():.4f} "
        f"std={rl_rewards.std():.4f}"
    )

    if base_rewards is not None:
        print(
            f">>> {args.baseline.capitalize()}: "
            f"mean={base_rewards.mean():.4f} "
            f"std={base_rewards.std():.4f}"
        )

    if args.save_plot:
        plot_results(base_rewards, rl_rewards)


if __name__ == "__main__":
    main()