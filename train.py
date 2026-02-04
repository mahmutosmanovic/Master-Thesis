from settings import *


def train_script(EPISODES, ROLLOUT_EPS, agent, model_path, animal, drone, logger, steps):
    run = None
    try:
        run = neptune.init_run(
            project=os.environ.get("NEPTUNE_PROJECT"),
            api_token=os.environ.get("NEPTUNE_API_TOKEN"),
        )
    except Exception as e:
        print(f"[WARN] Neptune not initialized: {e}")

    # Log settings
    if run is not None:
        run["settings/STEPS"].append(steps)
        run["settings/EPISODES"].append(EPISODES)
        run["settings/ROLLOUT_EPS"].append(ROLLOUT_EPS)
        run["settings/BEHAVIOR"].append(BEHAVIOR)
        run["settings/obs_dim"].append(obs_dim)
        run["settings/act_dim"].append(act_dim)
        run["settings/learning_rate"].append(learning_rate)

    # Reward tracking
    reward_history = []
    reward_queue = deque(maxlen=50)

    # Training loop
    for ep in range(1, EPISODES + 1):

        # periodic checkpoint
        if ep % 20 == 0:
            agent.save(model_path)

        ep_reward, ep_monitor, ep_disturb = agent.rollout_episode(
            animal,
            drone,
            logger,
            ep=ep,
            steps=steps,
            train=True,
        )

        agent.update(epochs=ROLLOUT_EPS)

        # rolling average
        reward_queue.append(ep_reward)
        avg_rew_50 = sum(reward_queue) / len(reward_queue)

        # Neptune logging
        if run is not None:
            run["train/episode_reward"].append(ep_reward)
            run["train/avg_reward_50"].append(avg_rew_50)
            run["train/monitor_reward"].append(ep_monitor)
            run["train/disturbance_penalty"].append(ep_disturb)

        reward_history.append(ep_reward)

        # Console log
        print(
            f"Episode {ep:4d} | "
            f"STEPS: {ep * steps:7d} | "
            f"TOT: {ep_reward:9.3f} | "
            f"AVG50: {avg_rew_50:9.3f} | "
            f"MON: {ep_monitor:9.3f} | "
            f"DIST_PEN: {ep_disturb:9.3f}"
        )

    # Cleanup
    if run is not None:
        run.stop()

    agent.save(model_path)

    return reward_history
