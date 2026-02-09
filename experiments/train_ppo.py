import argparse
import os
import numpy as np
import torch

# logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model.PPO import PPO, RolloutBuffer
from utils.utils import decode_action, log_config_text
from environment import Environment, EnvConfig, DroneParams, AnimalParams
from experiments.settings import rand5jack1drone


def train(
    env: Environment,
    total_steps=300_000,
    rollout_T=2048,
    update_epochs=10,
    batch_size=64,
    episode_horizon=500,
    device="cuda",
    seed=42,
    log_dir="logs/training",
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

    run_name = f"ppo_drone_seed{seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_log_dir)

    log_config_text(writer, cfg)

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
                scalars = env.episode_statistics()
                scalars["episode/return"] = float(ep_ret)
                scalars["episode/len"] = float(ep_len)

                for k, v in scalars.items():
                    writer.add_scalar(k, v, global_step=steps)

                writer.flush()
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
        if log_dir:
            save_dir = os.path.join(log_dir, "checkpoints", "ppo_drone.pt") # join with tb logging later
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            torch.save(
                {
                    "ac_state_dict": agent.ac.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
                save_dir,
            )

    writer.close()

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
    parser.add_argument("--log_dir", type=str, default="logs/training")
    args = parser.parse_args()

    cfg = rand5jack1drone()

    env = Environment(config=cfg)

    train(
        env=env,
        total_steps=args.steps,
        rollout_T=args.rollout,
        update_epochs=args.epochs,
        batch_size=args.batch,
        episode_horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
    )