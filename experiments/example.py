import numpy as np
from environment import Environment, decode_action
from .settings import standard_env

def random_drone_policy(action_dim):
    actions = {}
    for drone_id, action_dim in action_dim.items():
        action = np.random.uniform(-1, 1, size=action_dim)
        actions[drone_id] = decode_action(action)
    return actions

if __name__ == "__main__":
    cfg = standard_env()
    env = Environment(cfg)

    obs, info = env.reset(seed=42)
    print("--- Environment information ---")
    print(f"info on reset:")
    for key, val in info.items(): print(f"{key}: {val}")
    print(f"Observation dimensions by agent:\n {env.obs_dim}")
    print(f"Action dimensions by agent:\n {env.action_dim}")
    print(f"Global state dimension:\n {env.global_state_dim}")
    print()

    done = False
    total_reward = {drone_id: 0.0 for drone_id in env.drone_ids}

    while not done:
        actions = random_drone_policy(env.action_dim)
        obs, rewards, done, _ = env.step(actions)

        for drone_id, r in rewards.items():
            total_reward[drone_id] += r
        
        if done:
            print("--- Episode stats ---")
            print("Total rewards:", total_reward)
            env_stats = env.episode_statistics()
            print(f"episode_statistics:")
            for key, val in env_stats.items(): print(f"{key}: {val}")
            print()
            # env.render_episode(save_path="recordings/example_episode.mp4")
            # env.reset()


    print("--- Episode finished, saving episode recording ---")

    # render and save video
    env.save_log_csv("logs/simulations/example_rollout.csv")
    env.render_episode(save_path="recordings/example_episode.mp4")