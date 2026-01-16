import csv
import math
from collections import defaultdict

import pyray as pr # type: ignore


# CONFIG
CSV_PATH = "logs/simulations/test_jackal_random_single.csv"

SCREEN_W = 1280
SCREEN_H = 800
FPS = 60

WORLD_SCALE = 5.0      # meters -> pixels
AGENT_SIZE = 1.5       # visual size


# LOAD CSV
def load_csv(path):
    frames = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t"])
            frames[t].append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "vx": float(row["vx"]),
                "vy": float(row["vy"]),
                "species": row["species"],
            })

    times = sorted(frames.keys())
    return frames, times

# HELPERS
def sim_to_world(x, y, z):
    return pr.Vector3(
        x * WORLD_SCALE,
        z * WORLD_SCALE,
        y * WORLD_SCALE,
    )


def heading_from_velocity(vx, vy):
    return math.atan2(vy, vx)


# MAIN
def main():
    frames, times = load_csv(CSV_PATH)
    frame_idx = 0

    pr.init_window(SCREEN_W, SCREEN_H, "Stealth-Fleet Replay (Simple)")
    pr.set_target_fps(FPS)

    camera = pr.Camera3D(
        pr.Vector3(0, 120, 120),   # position
        pr.Vector3(0, 0, 0),       # target
        pr.Vector3(0, 1, 0),       # up
        45.0,                      # fovy
        pr.CAMERA_PERSPECTIVE      # projection
    )

    while not pr.window_should_close():
        frame = frames[times[frame_idx]]

        pr.begin_drawing()
        pr.clear_background(pr.SKYBLUE)

        pr.begin_mode_3d(camera)

        #  floor 
        pr.draw_plane(
            pr.Vector3(0, 0, 0),
            pr.Vector2(1000, 1000),
            pr.LIGHTGRAY
        )

        #  agents 
        for agent in frame:
            pos = sim_to_world(agent["x"], agent["y"], agent["z"])
            heading = heading_from_velocity(agent["vx"], agent["vy"])

            # body
            pr.draw_cylinder(
                pr.Vector3(pos.x, pos.y + 0.5, pos.z),
                0.5,
                0.5,
                1.0,
                12,
                pr.BROWN
            )

            # direction arrow (heading)
            dir_vec = pr.Vector3(
                math.cos(heading) * AGENT_SIZE,
                0,
                math.sin(heading) * AGENT_SIZE
            )

            pr.draw_line_3d(
                pr.Vector3(pos.x, pos.y + 1.0, pos.z),
                pr.Vector3(pos.x + dir_vec.x, pos.y + 1.0, pos.z + dir_vec.z),
                pr.RED
            )

        pr.end_mode_3d()

        pr.draw_text(
            f"Frame {frame_idx}/{len(times)}",
            10, 10, 20, pr.BLACK
        )

        pr.end_drawing()

        frame_idx = (frame_idx + 1) % len(times)

    pr.close_window()


if __name__ == "__main__":
    main()
