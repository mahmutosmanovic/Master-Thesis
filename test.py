from settings import *

def test_script(agent, model_path, animal, drone, logger, steps, CSV_PATH, draw_trail_3D):
        agent.load(model_path)

        # Run exactly 1 episode for visualization
        ep_reward, ep_monitor, ep_disturb = agent.rollout_episode(
            animal, drone, logger, ep=1, steps=steps, train=False
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
