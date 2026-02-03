from view import *
from agent import *
from drone import *
from utils import *
from logger import *
from pigeon import *
from reward import *
from settings import *

def rollout_episode(pigeon, drone, agent, logger, ep=1, steps=STEPS, train=False):
    """
    Runs one episode. If train=False uses deterministic policy for nicer playback.
    Logs to CSV via Logger.
    Returns: total_reward, total_monitor, total_disturb_pen
    """
    pigeon.reset()
    drone.reset((pigeon.x, pigeon.y, pigeon.z))

    obs = drone.observe(pigeon)

    ep_reward = 0.0
    ep_monitor = 0.0
    ep_disturb = 0.0

    for t in range(1, steps + 1):
        if train:
            action, logp, val = agent.act(obs)
        else:
            action = agent.act_deterministic(obs)
            # dummy for buffer (not used)
            logp, val = None, None

        dx, dy, dz, dyaw = action

        # Apply action (keep your current simple integration)
        drone.x += dx
        drone.y += dy
        drone.z += dz
        drone.yaw += dyaw * 0.1

        # (optional safety clamp)
        drone.z = max(0.0, drone.z)

        pigeon.step()

        next_obs = drone.observe(pigeon)

        # Reward
        animal_pos = (pigeon.x, pigeon.y, pigeon.z)
        drone_pos  = (drone.x, drone.y, drone.z)

        reward, monitoring_r, disturbance_pen = compute_total_reward(
            next_obs, animal_pos, drone_pos
        )

        done = (t == steps)

        # Store only in training mode
        if train:
            agent.store(obs, action, logp, val, reward, done)

        # Logging (same schema you created)
        step_id = (ep - 1) * steps + (t - 1)

        logger.write(
            CSV_PATH,
            ep,
            step_id,
            "PIGEON",
            (pigeon.x, pigeon.y, pigeon.z, 0),
            0,
            0,
            0,
        )

        logger.write(
            CSV_PATH,
            ep,
            step_id,
            "DRONE",
            (drone.x, drone.y, drone.z, drone.yaw),
            reward,
            monitoring_r,
            disturbance_pen,
        )

        ep_reward += reward
        ep_monitor += monitoring_r
        ep_disturb += disturbance_pen

        obs = next_obs

    return ep_reward, ep_monitor, ep_disturb


def main(train=False, run=False, model_path="ppo_drone.pt", steps=STEPS):
    pigeon_config = PigeonConfig()
    drone_config = DroneConfig()

    pigeon = Pigeon(pigeon_config, BEHAVIOR, start_pos=(0, 0, 0))
    drone = Drone(drone_config, (pigeon.x, pigeon.y, pigeon.z))

    agent = PPOAgent(obs_dim, act_dim, lr=learning_rate)

    logger = Logger()

    if train:
        reward_history = []
        for ep in range(1, EPISODES + 1):
            if ep % 20 == 0:
                agent.save(model_path)

            ep_reward, ep_monitor, ep_disturb = rollout_episode(
                pigeon, drone, agent, logger, ep=ep, steps=steps, train=True
            )

            agent.update()

            reward_history.append(ep_reward)
            print(
                f"Episode {ep} | STEPS: {ep*steps} | "
                f"TOT: {ep_reward:.3f} | MON: {ep_monitor:.3f} | DIST_PEN: {ep_disturb:.3f}"
            )

        agent.save(model_path)

    elif run:
        agent.load(model_path)

        # Run exactly 1 episode for visualization
        ep_reward, ep_monitor, ep_disturb = rollout_episode(
            pigeon, drone, agent, logger, ep=1, steps=steps, train=False
        )

        print(
            f"[RUN] TOT: {ep_reward:.3f} | MON: {ep_monitor:.3f} | DIST_PEN: {ep_disturb:.3f}"
        )

        # Load the CSV and visualize
        df = pd.read_csv(CSV_PATH)

        # If CSV contains multiple episodes, keep episode 1
        if "episode" in df.columns:
            df = df[df["episode"] == 1].reset_index(drop=True)

        draw_trail_3D(df, interval=50, trail_length=50)

    else:
        raise ValueError("Choose one: --train or --run")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--run", action="store_true")

    parser.add_argument("--model", type=str, default="ppo_drone.pt")
    parser.add_argument("--steps", type=int, default=STEPS)

    args = parser.parse_args()

    main(train=args.train, run=args.run, model_path=args.model, steps=args.steps)
