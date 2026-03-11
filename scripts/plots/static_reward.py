import os
import numpy as np
import matplotlib.pyplot as plt

from environment.env import Env
from environment.vec import Vector
from config.loader import load_config
from box import Box


def sync_animal_states_from_disturbance(env):
    """
    Make static plotting use the same calm/avoid/flee thresholds
    as the env behavior logic, without actually moving animals.
    """
    for animal in env.animals:
        D = animal.disturbance

        if D > 0.70:
            animal.state = "flee"
        elif D > 0.50:
            animal.state = "avoid"
        else:
            animal.state = "calm"

def place_static_scene(env, radial_distance, z, azimuth_deg=0.0):
    """
    Assumes exactly 1 animal and 1 drone.
    Animal is fixed at origin, drone at (r, azimuth, z),
    and camera points directly at the animal.
    """
    if env.animal_count != 1 or env.drone_count != 1:
        raise ValueError("This script assumes exactly 1 animal and 1 drone.")

    animal = env.animals[0]
    drone = env.drones[0]

    animal_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    animal.pos.setter(Vector(*animal_pos))
    animal.vel_dir.setter(Vector(1.0, 0.0, 0.0))
    animal.vel_speed = 0.0
    animal.disturbance = 0.0
    animal.escape_dir = np.zeros(3, dtype=np.float32)
    animal.state = "calm"

    phi = np.deg2rad(azimuth_deg)
    drone_pos = np.array([
        radial_distance * np.cos(phi),
        radial_distance * np.sin(phi),
        z,
    ], dtype=np.float32)
    drone.pos.setter(Vector(*drone_pos))

    look_vec = animal_pos - drone_pos
    norm = np.linalg.norm(look_vec)
    if norm > 1e-8:
        look_vec = look_vec / norm
    else:
        look_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    drone.view_dir.setter(Vector(*look_vec))
    drone.view_dir.unit()

    # Static drone
    # Keep velocity zero so heading/speed disturbance contributions vanish.
    drone.vel_dir.setter(Vector(*look_vec))
    drone.vel_dir.unit()
    drone.vel_speed = 0.0
    drone.theta = 0.0


def compute_total_reward_exact(env):
    """
    Uses the actual env pipeline and actual Env.compute_reward().
    Protects reward_stats from being polluted by the sweep.
    """
    geometry = env._compute_geometry()
    env._compute_disturbance(geometry)
    observations = env._build_observations(geometry)

    reward_stats_backup = env.reward_stats.copy()
    total_state_steps_backup = env.total_state_steps

    reward = env.compute_reward(observations)

    env.reward_stats = reward_stats_backup
    env.total_state_steps = total_state_steps_backup

    return reward


def evaluate_reward_grid(env, r_values, z_values, azimuth_deg=0.0):
    reward_grid = np.zeros((len(z_values), len(r_values)), dtype=np.float32)

    for iz, z in enumerate(z_values):
        for ir, r in enumerate(r_values):
            place_static_scene(env, r, z, azimuth_deg=azimuth_deg)

            geometry = env._compute_geometry()
            env._compute_disturbance(geometry)

            # crucial line for your state-based reward
            sync_animal_states_from_disturbance(env)

            observations = env._build_observations(geometry)
            reward_grid[iz, ir] = env.compute_reward(observations)

    return reward_grid


def plot_static_total_reward(env, r_values=None, z_values=None, azimuth_deg=0.0):
    if env.animal_count != 1 or env.drone_count != 1:
        raise ValueError("Use exactly 1 animal and 1 drone.")

    if r_values is None:
        r_values = np.linspace(0.0, 150.0, 200)

    if z_values is None:
        z_values = np.linspace(0.0, 150.0, 200)

    reward_grid = evaluate_reward_grid(env, r_values, z_values, azimuth_deg=azimuth_deg)

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        reward_grid,
        origin="lower",
        aspect="auto",
        extent=[r_values[0], r_values[-1], z_values[0], z_values[-1]],
    )
    plt.colorbar(label="Total reward")
    plt.xlabel("Radial distance")
    plt.ylabel("z")
    plt.title("Static total reward")
    plt.tight_layout()
    plt.savefig("figures/static_reward.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    cfg = Box(load_config("train"))
    env = Env(cfg, render_mode=None, seed=42)
    env.reset(seed=42)

    plot_static_total_reward(env)