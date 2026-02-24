import json
import argparse
import itertools
import numpy as np
from box import Box
from environment import Env
from config import load_config

def build_drone_specs(config_box):
    specs = []

    # Must match Env ordering exactly
    for drone_type in config_box.drone:
        drone_config = config_box.drone[drone_type]
        count = int(drone_config.count)

        for _ in range(count):
            specs.append({
                "view_range": drone_config.view_range,
                "max_altitude": drone_config.max_altitude,
                "ver_angle": drone_config.ver_angle,
                "hor_angle": drone_config.hor_angle,
            })

    return specs

class CentroidStandoff:
    def __init__(
        self,
        config,
        target_range_ratio=0.45,
        target_altitude_ratio=0.55,
        xy_gain=1.0,
        z_gain=1.0,
        theta_gain=0.8,
        search_theta=0.35,
        min_speed_norm=0.0,
    ):
        self.config = config
        self.target_range_ratio = target_range_ratio
        self.target_altitude_ratio = target_altitude_ratio
        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.theta_gain = theta_gain
        self.search_theta = search_theta
        self.min_speed_norm = min_speed_norm

        self.drone_specs = build_drone_specs(config_box)
        self.drone_count = len(self.drone_specs)

    def _unit(self, v, eps=1e-8):
        n = np.linalg.norm(v)
        return v / (n + eps)

    def _camera_basis_from_view_dir(self, view_dir):
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        x = self._unit(view_dir.astype(np.float32))

        y = np.cross(world_z, x)
        y = self._unit(y)

        z = np.cross(x, y)
        z = self._unit(z)

        return x, y, z

    def _cam_vector_from_angles(self, h_angle, v_angle):
        th = np.tan(h_angle)
        tv = np.tan(v_angle)

        cx = 1.0 / np.sqrt(1.0 + th * th + tv * tv)
        cy = th * cx
        cz = tv * cx
        return np.array([cx, cy, cz], dtype=np.float32)

    def act(self, observations):
        actions = np.zeros((self.drone_count, 5), dtype=np.float32)

        for d in range(self.drone_count):
            drone = self.drone_specs[d]
            max_altitude = drone["max_altitude"]
            view_range = drone["view_range"]
            ver_angle = drone["ver_angle"]
            hor_angle = drone["hor_angle"]
            
            obs_d = observations[d]

            in_view = obs_d[:, 0] > 0.5
            visible_idx = np.where(in_view)[0]

            view_dir = obs_d[0, 4:7].astype(np.float32)
            altitude_norm = obs_d[0, 7]
            current_altitude = altitude_norm * (max_altitude + 1e-8)

            x, y, z = self._camera_basis_from_view_dir(view_dir)

            if len(visible_idx) == 0:
                # with no targets in view, spin
                target_altitude = self.target_altitude_ratio * max_altitude
                z_error = target_altitude - current_altitude
                z_action = np.clip(
                    self.z_gain * (z_error / max(max_altitude, 1e-6)),
                    -1.0,
                    1.0,
                )

                move_dir = x + np.array([0.0, 0.0, z_action], dtype=np.float32)
                move_dir = self._unit(move_dir)

                actions[d] = np.array([
                    move_dir[0], move_dir[1], move_dir[2],
                    self.min_speed_norm,
                    self.search_theta,
                ], dtype=np.float32)
                continue

            # Reconstruct relative vectors to visible animals
            rel_vecs = []
            h_norms = []

            v_max = np.deg2rad(ver_angle / 2.0)
            h_max = np.deg2rad(hor_angle / 2.0)

            for a in visible_idx:
                row = obs_d[a]
                dist_norm = row[1]
                v_norm = row[2]
                h_norm = row[3]

                distance = dist_norm * view_range
                v_angle = v_norm * v_max
                h_angle = h_norm * h_max

                # animal in camera vector -> world vector
                cam_vector = self._cam_vector_from_angles(h_angle, v_angle)
                world_vec = cam_vector[0] * x + cam_vector[1] * y + cam_vector[2] * z
                world_vec = self._unit(world_vec)

                rel_vec = distance * world_vec
                rel_vecs.append(rel_vec)
                h_norms.append(h_norm)

            rel_vecs = np.asarray(rel_vecs, dtype=np.float32)
            rel_centroid = rel_vecs.mean(axis=0)

            # xy standoff
            centroid_xy = rel_centroid[:2]
            centroid_xy_norm = np.linalg.norm(centroid_xy)

            dir_to_centroid_xy = centroid_xy / centroid_xy_norm

            target_range = self.target_range_ratio * view_range
            xy_error = centroid_xy_norm - target_range

            xy_action = np.clip(
                self.xy_gain * (xy_error / max(view_range, 1e-6)),
                -1.0,
                1.0,
            )

            # z control
            target_altitude = self.target_altitude_ratio * max_altitude
            z_error = target_altitude - current_altitude
            z_action = np.clip(
                self.z_gain * (z_error / max(max_altitude, 1e-6)),
                -1.0,
                1.0,
            )

            # full movement direction
            move_xy = xy_action * dir_to_centroid_xy
            move_dir = np.array([move_xy[0], move_xy[1], z_action], dtype=np.float32)
            move_dir = self._unit(move_dir)

            # speed control
            speed_effort = 0.7 * abs(xy_action) + 0.3 * abs(z_action)
            norm_speed = np.clip(self.min_speed_norm + speed_effort, 0.0, 1.0)
            h_center = np.mean(h_norms) if len(h_norms) > 0 else 0.0
            norm_theta = -np.clip(-self.theta_gain * h_center, -1.0, 1.0)

            actions[d] = np.array([
                move_dir[0], move_dir[1], move_dir[2],
                norm_speed,
                norm_theta,
            ], dtype=np.float32)

        return actions

def run_episode(env, policy, seed):
    obs, info = env.reset(seed)

    terminated = False
    truncated = False

    step_count = 0
    episode_reward = 0.0

    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        episode_reward += float(reward)

    norm_reward = episode_reward / env.config.max_episode_steps
    stats = env.get_behavior_stats()
    return norm_reward, step_count, stats

GRID = {
    "target_range_ratio":    [0.3, 0.35, 0.40, 0.45],
    "target_altitude_ratio": [0.4, 0.45, 0.5],
    "xy_gain":                   [1.25, 1.5, 1.75],
    "z_gain":                   [1.25, 1.5, 1.75],
    "theta_gain":               [0.4],
    "search_theta":          [0.35],
    "min_speed_norm":        [0.15],
}

def evaluate_params(env, params, seeds):
    rewards = []

    for seed in seeds:
        # Recreate policy each episode (avoids internal-state carryover)
        policy = CentroidStandoff(env, **params)
        r, steps, stats = run_episode(env, policy, seed=int(seed))
        rewards.append(r)

    rewards = np.asarray(rewards, dtype=np.float32)
    return np.mean(rewards), np.std(rewards), rewards.tolist()


def grid_search(config_box, args):
    env = Env(config_box, render_mode=None)

    # Fixed seed set for fair comparison (deterministic if env seeding is deterministic)
    seeds = [args.seed + i for i in range(args.eval_seeds)]

    keys = list(GRID.keys())
    value_lists = [GRID[k] for k in keys]

    total = 1
    for vals in value_lists:
        total *= len(vals)

    print(f"Grid search over {total} combinations")
    print(f"Eval seeds: {seeds}")

    best_mean = -np.inf
    best_std = None
    best_params = None

    for i, values in enumerate(itertools.product(*value_lists)):
        params = dict(zip(keys, values))

        mean_r, std_r, rewards = evaluate_params(env, params, seeds)

        improved = mean_r > best_mean
        if improved:
            best_mean = mean_r
            best_std = std_r
            best_params = params

        mark = "*" if improved else " "
        print(
            f"[{i+1:04d}/{total}] {mark} "
            f"mean={mean_r:.4f} std={std_r:.4f} params={params}"
        )

    print("\n=== GRID SEARCH RESULTS ===")
    print(f"Best mean reward: {best_mean:.4f}")
    print(f"Best std reward:  {best_std:.4f}")
    print("Best params:", json.dumps(best_params, indent=2))

    if args.render_best:
        print("\nRendering best policy...")
        render_env = Env(config_box, render_mode="human")
        policy = CentroidStandoff(render_env, **best_params)
        r, steps, stats = run_episode(render_env, policy, seed=int(args.seed))
        print(f"Rendered on seed {args.seed} | norm reward={r:.4f}")
        if stats is not None:
            print("Behavior stats:", stats)
        if hasattr(render_env, "viewer") and render_env.viewer is not None:
            render_env.viewer.close()

    if hasattr(env, "viewer") and env.viewer is not None:
        try:
            env.viewer.close()
        except Exception:
            pass

def run_single(config_box, seed):
    env = Env(config_box, render_mode="human")
    policy = CentroidStandoff(
        env,
        target_range_ratio=0.45,
        target_altitude_ratio=0.45,
        xy_gain=1.5,
        z_gain=1.25,
        theta_gain=0.4,
        search_theta=0.35,
        min_speed_norm=0.15,
    )

    norm_reward, steps, stats = run_episode(env, policy, seed=seed)
    print(f"Episode finished. Total Reward: {norm_reward:.4f}")
    if stats is not None:
        print("Behavior stats:", stats)

    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.close()


# -------------------------
# CLI
# -------------------------

def _init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="Config name inside config/ folder",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed (also first eval seed in grid mode)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "grid"],
        help="run = single baseline episode, grid = basic grid search",
    )

    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=3,
        help="How many seeds to average per grid point",
    )

    parser.add_argument(
        "--render-best",
        action="store_true",
        help="Render best found params after grid search",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _init_argparse()

    cfg = load_config(args.config)
    config_box = Box(cfg)

    if args.mode == "run":
        run_single(config_box, seed=args.seed)
    else:
        grid_search(config_box, args)