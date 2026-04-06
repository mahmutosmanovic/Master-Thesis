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
        target_range_ratio=0.3,
        target_altitude_ratio=0.5,
        xy_gain=1.6,
        z_gain=0.5,
        theta_gain=0.4,
        search_theta=0.25,
        xy_deadband=0.01,
        z_deadband=0.01,
        theta_deadband=0.05,
        max_speed_norm=1.0,
        search_turn_only=True,
    ):
        self.config = config

        self.target_range_ratio = target_range_ratio
        self.target_altitude_ratio = target_altitude_ratio

        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.theta_gain = theta_gain
        self.search_theta = search_theta

        self.xy_deadband = xy_deadband
        self.z_deadband = z_deadband
        self.theta_deadband = theta_deadband

        self.max_speed_norm = max_speed_norm
        self.search_turn_only = search_turn_only

        self.drone_specs = build_drone_specs(self.config)
        self.drone_count = len(self.drone_specs)

    def _unit(self, v, eps=1e-8):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)

    def _camera_basis_from_view_dir(self, view_dir):
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        x = self._unit(view_dir)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        y = np.cross(world_z, x)
        if np.linalg.norm(y) < 1e-6:
            # view_dir nearly parallel to world_z
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(x, fallback)) > 0.95:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            y = np.cross(fallback, x)

        y = self._unit(y)
        z = self._unit(np.cross(x, y))

        return x, y, z

    def _cam_vector_from_angles(self, h_angle, v_angle):
        th = np.tan(h_angle)
        tv = np.tan(v_angle)

        cx = 1.0 / np.sqrt(1.0 + th * th + tv * tv)
        cy = th * cx
        cz = tv * cx
        return np.array([cx, cy, cz], dtype=np.float32)

    def _p_control(self, error_norm, gain, deadband):
        if abs(error_norm) < deadband:
            return 0.0
        return float(np.clip(gain * error_norm, -1.0, 1.0))

    def _make_motion_command(self, move_vec):
        move_vec = np.asarray(move_vec, dtype=np.float32)
        move_mag = float(np.linalg.norm(move_vec))

        if move_mag < 1e-6:
            move_dir = np.zeros(3, dtype=np.float32)
            norm_speed = 0.0
        else:
            move_dir = (move_vec / move_mag).astype(np.float32)
            norm_speed = float(np.clip(move_mag, 0.0, self.max_speed_norm))

        return move_dir, norm_speed

    def _safe_xy_direction(self, vec_xy, fallback_xy):
        vec_xy = np.asarray(vec_xy, dtype=np.float32)
        n = np.linalg.norm(vec_xy)
        if n < 1e-6:
            fb = self._unit(fallback_xy)
            if np.linalg.norm(fb) < 1e-6:
                fb = np.array([1.0, 0.0], dtype=np.float32)
            return fb
        return (vec_xy / n).astype(np.float32)

    def act(self, observations):
        n_actions = self.config.model.space.n_actions
        actions = np.zeros((self.drone_count, n_actions), dtype=np.float32)

        d_feats = self.config.model.space.drone_features
        a_feats = self.config.model.space.animal_features
        n_a = self.config.animal.env.count

        for d in range(self.drone_count):
            drone = self.drone_specs[d]
            max_altitude = float(drone["max_altitude"])
            view_range = float(drone["view_range"])
            ver_angle = float(drone["ver_angle"])
            hor_angle = float(drone["hor_angle"])

            obs_d = observations[d]

            # --- split observation ---
            drone_features = obs_d[:d_feats]
            animal_obs = obs_d[d_feats:d_feats + n_a * a_feats].reshape(n_a, a_feats)

            view_dir = np.asarray(drone_features[:3], dtype=np.float32)
            altitude_norm = float(drone_features[3])
            current_altitude = altitude_norm * (max_altitude + 1e-8)

            in_view = animal_obs[:, 0] > 0.5
            is_target = animal_obs[:, 7] > 0.5
            visible_idx = np.where(in_view & is_target)[0]

            x, y, z = self._camera_basis_from_view_dir(view_dir)

            # --- altitude hold ---
            target_altitude = self.target_altitude_ratio * max_altitude
            z_error_norm = (target_altitude - current_altitude) / max(max_altitude, 1e-6)
            z_cmd = self._p_control(z_error_norm, self.z_gain, self.z_deadband)

            # ------------------------------------------------------------------
            # SEARCH MODE: no visible target
            # Rotate in place. Only move vertically if altitude needs correction.
            # ------------------------------------------------------------------
            if len(visible_idx) == 0:
                if self.search_turn_only:
                    move_vec = np.array([0.0, 0.0, z_cmd], dtype=np.float32)
                else:
                    # Optional: slow forward search if you ever want it
                    move_vec = np.array([0.15, 0.0, z_cmd], dtype=np.float32)

                move_dir, norm_speed = self._make_motion_command(move_vec)

                actions[d] = np.array([
                    move_dir[0],
                    move_dir[1],
                    move_dir[2],
                    norm_speed,
                    self.search_theta,
                ], dtype=np.float32)
                continue

            # ------------------------------------------------------------------
            # TRACKING MODE
            # ------------------------------------------------------------------
            rel_vecs = []
            h_norms = []

            v_max = np.deg2rad(ver_angle / 2.0)
            h_max = np.deg2rad(hor_angle / 2.0)

            for a in visible_idx:
                row = animal_obs[a]

                dist_norm = float(row[1])
                v_norm = float(row[2])
                h_norm = float(row[3])

                distance = dist_norm * view_range
                v_angle = v_norm * v_max
                h_angle = h_norm * h_max

                cam_vector = self._cam_vector_from_angles(h_angle, v_angle)
                world_vec = cam_vector[0] * x + cam_vector[1] * y + cam_vector[2] * z
                world_vec = self._unit(world_vec)

                rel_vec = distance * world_vec
                rel_vecs.append(rel_vec)
                h_norms.append(h_norm)

            if len(rel_vecs) == 0:
                # Defensive fallback: hover + no yaw
                actions[d] = np.zeros(n_actions, dtype=np.float32)
                continue

            rel_vecs = np.asarray(rel_vecs, dtype=np.float32)
            rel_centroid = rel_vecs.mean(axis=0)

            centroid_xy = rel_centroid[:2]
            centroid_xy_norm = float(np.linalg.norm(centroid_xy))

            # Desired horizontal standoff
            target_range = self.target_range_ratio * view_range
            xy_error_norm = (centroid_xy_norm - target_range) / max(view_range, 1e-6)
            xy_cmd = self._p_control(xy_error_norm, self.xy_gain, self.xy_deadband)

            # If centroid_xy is almost zero, pick a stable fallback horizontal direction
            forward_xy = x[:2]
            dir_to_centroid_xy = self._safe_xy_direction(centroid_xy, fallback_xy=forward_xy)

            # Translation vector in world frame
            move_vec = np.array([
                xy_cmd * dir_to_centroid_xy[0],
                xy_cmd * dir_to_centroid_xy[1],
                z_cmd,
            ], dtype=np.float32)

            move_dir, norm_speed = self._make_motion_command(move_vec)

            # Yaw control from image horizontal offset
            h_center = float(np.mean(h_norms)) if len(h_norms) > 0 else 0.0
            theta_cmd = self._p_control(h_center, self.theta_gain, self.theta_deadband)

            actions[d] = np.array([
                move_dir[0],
                move_dir[1],
                move_dir[2],
                norm_speed,
                theta_cmd,
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
    "target_range_ratio":    [0.2, 0.3, 0.4, 0.5, 0.6],
    "target_altitude_ratio": [0.2, 0.3, 0.4, 0.5, 0.6],
    "xy_gain":               [0.8, 1.2, 1.6],
    "z_gain":                [0.5, 0.8, 1.1],
    "theta_gain":            [0.4],
    "search_theta":          [0.25],
    "xy_deadband":           [0.02],
    "z_deadband":            [0.02],
    "theta_deadband":        [0.03],
}


def evaluate_params(env, params, seeds):
    rewards = []

    for seed in seeds:
        policy = CentroidStandoff(env.config, **params)
        r, steps, stats = run_episode(env, policy, seed=int(seed))
        rewards.append(r)

    rewards = np.asarray(rewards, dtype=np.float32)
    return np.mean(rewards), np.std(rewards), rewards.tolist()


def grid_search(config_box, args):
    env = Env(config_box, render_mode=None)

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
        policy = CentroidStandoff(render_env.config, **best_params)
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

    policy = CentroidStandoff(env.config)

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
    config_box.model.space.action_type = "abs"

    if args.mode == "run":
        run_single(config_box, seed=args.seed)
    else:
        grid_search(config_box, args)