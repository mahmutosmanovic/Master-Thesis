# experiments/run_ppo.py
import os
import argparse
import numpy as np
import torch

from utils.vec_utils import unit
from environment.environment import Environment
from model.model import PPO

def decode_action(a: np.ndarray):
    a = np.asarray(a, dtype=np.float32)

    direction = unit(a[:3])

    # speed: [0,1]
    speed = float((a[3] + 1.0) * 0.5)

    # yaw rate: [-1,1]
    view_yaw_rate = float(a[4])

    return (direction, speed, view_yaw_rate)


@torch.no_grad()
def act_deterministic(agent: PPO, obs: np.ndarray):
    """
    Deterministic PPO action: tanh(mu). (No sampling noise.)
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    mu, std, v = agent.ac(obs_t)
    a = torch.tanh(mu).squeeze(0).cpu().numpy().astype(np.float32)
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ppo_drone.pt")
    parser.add_argument("--log", type=str, default="logs/simulations/ppo_rollout.csv")
    parser.add_argument("--print_every", type=int, default=1)
    args = parser.parse_args()

    # --- Env ---
    env = Environment(seed=args.seed)
    obs_dict, info = env.reset(seed=args.seed)
    drone_ids = info["drone_ids"]

    # --- Load checkpoint + build PPO ---
    ckpt = torch.load(args.ckpt, map_location=args.device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    agent = PPO(obs_dim, act_dim, device=args.device)
    agent.ac.load_state_dict(ckpt["ac_state_dict"])
    agent.ac.eval()

    # --- Rollout ---
    rewards_total = {int(did): 0.0 for did in drone_ids}

    for t in range(args.steps):
        external_actions = {}

        for did in drone_ids:
            did = int(did)
            obs = np.asarray(obs_dict[did], dtype=np.float32)

            a = act_deterministic(agent, obs)  # shape (5,)
            external_actions[did] = decode_action(a)
        obs_dict, reward_dict, done, info_step = env.step(external_actions)

        for did, r in reward_dict.items():
            rewards_total[int(did)] += float(r)

        if args.print_every > 0 and ((t + 1) % args.print_every == 0):
            print(f"step {t+1}/{args.steps} reward={reward_dict}")

        if done:
            print("done=True, stopping early")
            break

    print("total rewards:", rewards_total)

    # --- Save log ---
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    env.save_log_csv(args.log)
    print("saved log:", args.log)


if __name__ == "__main__":
    main()