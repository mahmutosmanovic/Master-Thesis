import argparse
import os
import numpy as np
import torch

from utils.vec_utils import unit
from model.model import PPO, RolloutBuffer  # your existing PPO + buffer
from environment.environment import Environment 

def decode_action(a: np.ndarray):
    a = np.asarray(a, dtype=np.float32)

    direction = unit(a[:3])

    # speed: [-1,1]
    speed = float((a[3] + 1.0) * 0.5)

    # yaw rate: [-1,1]
    view_yaw_rate = float(a[4])

    return (direction, speed, view_yaw_rate)

def train(
    env: Environment,
    total_steps=300_000,
    rollout_T=2048,
    update_epochs=10,
    batch_size=64,
    episode_horizon=500,
    device="cuda",
    seed=42,
    save_path="checkpoints/ppo_drone.pt",
):
    device_t = torch.device(device)

    obs_dict, info = env.reset(seed=seed)
    drone_ids = info["drone_ids"]
    train_id = drone_ids[0]

    obs = np.asarray(obs_dict[train_id], dtype=np.float32)
    obs_dim = int(obs.shape[0])
    act_dim = 5

    print(f"observation dim: {obs_dim}, action dim: {act_dim}")
    agent = PPO(obs_dim, act_dim, device=str(device_t))
    buf = RolloutBuffer(rollout_T, obs_dim, act_dim, device=device_t)

    ep_ret = 0.0
    ep_len = 0
    steps = 0

    while steps < total_steps:
        buf.reset()

        for _ in range(rollout_T):
            a, pre_tanh, logp, v = agent.act(obs)

            # Build action dict for all drones
            external_actions = {}
            for did in drone_ids:
                drone = env.agents[did]
                if did == train_id:
                    external_actions[did] = decode_action(a)
                else:
                    external_actions[did] = (
                        np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        0.0,
                        0.0,
                    )

            next_obs_dict, reward_dict, done_env, info_step = env.step(external_actions)

            next_obs = np.asarray(next_obs_dict[train_id], dtype=np.float32)
            r = float(reward_dict[train_id])

            ep_len += 1
            truncated = ep_len >= episode_horizon
            terminated = bool(done_env)  # your env currently always False
            done = terminated or truncated

            buf.add(
                torch.as_tensor(obs, dtype=torch.float32, device=device_t),
                torch.as_tensor(a, dtype=torch.float32, device=device_t),
                torch.as_tensor(pre_tanh, dtype=torch.float32, device=device_t), # Add this
                torch.as_tensor(logp, dtype=torch.float32, device=device_t),
                torch.as_tensor(r, dtype=torch.float32, device=device_t),
                torch.as_tensor(float(done), dtype=torch.float32, device=device_t),
                torch.as_tensor(v, dtype=torch.float32, device=device_t),
            )

            obs = next_obs
            ep_ret += r
            steps += 1

            if done:
                print(f"episode return={ep_ret:.2f} len={ep_len} steps={steps}")
                obs_dict, info = env.reset()
                drone_ids = info["drone_ids"]
                train_id = drone_ids[0]
                obs = np.asarray(obs_dict[train_id], dtype=np.float32)
                ep_ret = 0.0
                ep_len = 0

            if steps >= total_steps:
                break

        # Bootstrap value for GAE
        with torch.no_grad():
            _, _, last_val = agent.ac(torch.as_tensor(obs, dtype=torch.float32, device=device_t).unsqueeze(0))
            last_val = last_val.squeeze(0)

        agent.update(buf, last_val, epochs=update_epochs, batch_size=batch_size)

        # Periodic save (simple)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "ac_state_dict": agent.ac.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
                save_path,
            )

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default="checkpoints/ppo_drone.pt")
    args = parser.parse_args()

    env = Environment(seed=args.seed)

    train(
        env=env,
        total_steps=args.steps,
        rollout_T=args.rollout,
        update_epochs=args.epochs,
        batch_size=args.batch,
        episode_horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        save_path=args.save,
    )