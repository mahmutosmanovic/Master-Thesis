import json
import argparse
import itertools
import numpy as np
from box import Box
from environment import Env
from config import load_config

def build_drone_specs(config_box):
    specs = []

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
        target_range_ratio=0.2,
        target_altitude_ratio=0.5,
        forward_gain=1.6,
        up_gain=1.1,
        theta_gain=0.6,
        search_theta=0.25,
        search_forward=0.0,
        max_speed_norm=1.0,
    ):
        self.config = config

        self.target_range_ratio = target_range_ratio
        self.target_altitude_ratio = target_altitude_ratio

        self.forward_gain = forward_gain
        self.up_gain = up_gain
        self.theta_gain = theta_gain
        self.search_theta = search_theta
        self.search_forward = search_forward

        self.max_speed_norm = max_speed_norm

        self.drone_specs = build_drone_specs(self.config)
        self.drone_count = len(self.drone_specs)

    def _make_motion_command(self, move_vec):
        move_vec = np.asarray(move_vec, dtype=np.float32)
        move_mag = float(np.linalg.norm(move_vec))

        if move_mag < 1e-6:
            move_dir = np.zeros(3, dtype=np.float32)
            norm_speed = -1.0
        else:
            move_dir = (move_vec / (move_mag + 1e-8)).astype(np.float32)

            speed_frac = float(np.clip(move_mag, 0.0, self.max_speed_norm))
            speed_frac = speed_frac / max(self.max_speed_norm, 1e-8)   # 0..1
            norm_speed = 2.0 * speed_frac - 1.0                        # -> -1..1

        return move_dir, float(np.clip(norm_speed, -1.0, 1.0))

    def _soft_zone(self, x, width):
        x = float(x)
        width = max(float(width), 1e-8)

        # Smoothly suppress tiny errors, preserve larger ones.
        scale = 1.0 - np.exp(- (abs(x) / width) ** 2)
        return float(x * scale)
    
    def _p_control(self, error, gain, soft_zone=0.0):
        e = self._soft_zone(error, soft_zone) if soft_zone > 0.0 else float(error)
        return float(np.clip(gain * e, -1.0, 1.0))

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

            obs_d = observations[d]

            # --- split observation ---
            drone_features = obs_d[:d_feats]
            animal_obs = obs_d[d_feats:d_feats + n_a * a_feats].reshape(n_a, a_feats)

            altitude_norm = float(drone_features[3])
            current_altitude = altitude_norm * (max_altitude + 1e-8)

            in_view = animal_obs[:, 0] > 0.5
            is_target = animal_obs[:, 7] > 0.5
            visible_idx = np.where(in_view & is_target)[0]

            # ------------------------------------------------------------------
            # SEARCH MODE: no visible target
            # Keep search simple: optional forward drift + altitude hold + yaw scan.
            # ------------------------------------------------------------------
            if len(visible_idx) == 0:
                target_altitude = self.target_altitude_ratio * max_altitude
                altitude_error_norm = (target_altitude - current_altitude) / max(max_altitude, 1e-6)
                up_cmd = self._p_control(altitude_error_norm, self.up_gain)

                move_vec = np.array([
                    self.search_forward,
                    0.0,
                    up_cmd,
                ], dtype=np.float32)

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
            # Minimal rel policy:
            # - distance controls forward
            # - h controls right + yaw
            # - v controls up
            # ------------------------------------------------------------------
            rows = animal_obs[visible_idx]

            dist_center = float(np.mean(rows[:, 1]))
            v_center = float(np.mean(rows[:, 2]))
            h_center = float(np.mean(rows[:, 3]))

            target_dist_norm = float(np.clip(self.target_range_ratio, 0.0, 1.0))
            forward_error = dist_center - target_dist_norm

            forward_cmd = self._p_control(forward_error, self.forward_gain, 0.01)
            up_cmd = self._p_control(v_center, self.up_gain, 0.01)
            theta_cmd = self._p_control(h_center, self.theta_gain, 0.01)

            move_vec = np.array([
                forward_cmd,
                0.0,
                up_cmd,
            ], dtype=np.float32)

            move_dir, norm_speed = self._make_motion_command(move_vec)

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
    "target_range_ratio":    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "target_altitude_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "forward_gain":          [1.6],
    "up_gain":               [1.1],
    "theta_gain":            [0.6],
    "search_theta":          [0.25],
    "search_forward":        [0.0],
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
    config_box.model.space.action_type = "rel"

    if args.mode == "run":
        run_single(config_box, seed=args.seed)
    else:
        grid_search(config_box, args)
