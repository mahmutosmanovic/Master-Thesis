from settings import *

def train_script(EPISODES, ROLLOUT_EPS, agent, model_path, animal, drone, logger, steps):
        run = neptune.init_run(
            project=os.environ["NEPTUNE_PROJECT"],
            api_token=os.environ["NEPTUNE_API_TOKEN"],
        )

        if run is not None:
            run["settings/STEPS"].append(STEPS)
            run["settings/EPISODES"].append(EPISODES)
            run["settings/ROLLOUT_EPS"].append(ROLLOUT_EPS)
            run["settings/BEHAVIOR"].append(BEHAVIOR)
            run["settings/obs_dim"].append(obs_dim)
            run["settings/act_dim"].append(act_dim)
            run["settings/learning_rate"].append(learning_rate)

        reward_history = []
        for ep in range(1, EPISODES + 1):
            if ep % 20 == 0:
                agent.save(model_path)

            ep_reward, ep_monitor, ep_disturb = agent.rollout_episode(
                animal, drone, logger, ep=ep, steps=steps, train=True
            )

            agent.update(epochs=ROLLOUT_EPS)

            if run is not None:
                run["train/episode_reward"].append(ep_reward)
                run["train/monitor_reward"].append(ep_monitor)
                run["train/disturbance_penalty"].append(ep_disturb)


            reward_history.append(ep_reward)
            print(
                f"Episode {ep:4d} | "
                f"STEPS: {ep*steps:4d} | "
                f"TOT: {ep_reward:8.3f} | "
                f"MON: {ep_monitor:8.3f} | "
                f"DIST_PEN: {ep_disturb:8.3f}"
            )


        if run is not None:
            run.stop()

        agent.save(model_path)