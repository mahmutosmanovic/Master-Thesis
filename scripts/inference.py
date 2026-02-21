# config
from scripts.config import cfg_train

# environment
from environment import Env

# model
from model import Agent

# standard
from box import Box
import torch
import os


def main(config):

    config = Box(config)

    env = Env(config, render_mode=None, seed=99)
    obs, info = env.reset()

    agent = Agent(config)

    model_dir = "tmp/ppo"

    actor_path = os.path.join(model_dir, "actor_torch_ppo.pt")
    agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.actor.eval()

    env.set_render_mode("human")

    terminated = False
    truncated = False

    step_count = 0
    episode_reward = 0.0

    while not (terminated or truncated):

        with torch.no_grad():
            action, _, _ = agent.choose_action(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        episode_reward += reward

        env.viewer.draw(
            env.drones,
            env.animals,
            env.render_mode,
            fov=info["fov"],
            reward=reward,
        )
            
    norm_reward = episode_reward / config.max_episode_steps
    print(f"Episode finished. Total Reward: {norm_reward:.4f}")

    env.viewer.close()


if __name__ == "__main__":
    main(cfg_train)
