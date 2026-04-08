import numpy as np

from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .resource_map import ResourceMap
from .immutables import MovementDim
from .behaviors import CRW_CFG
from .disturbance import (
    disturbance_gain,
    interpolate_monitor_disturbance,
    rel_vec_to_xz,
    monitor_reward as disturbance_monitor_reward,
    raw_disturbance_calc,
)


class Env:
    def __init__(self, config, render_mode=None, seed=42):
        self.set_seed(seed)
        self.render_mode = render_mode
        self.config = config
        self.disturb_scale = config.disturbance_penalty_scale
        self.viewer = Viewer(config)
        self.resource_map = None
        self.lost_track_counter = 0
        self.spawn_radius = self.config["animal"]["init"]["max_spawn_radius"]

        self.animal_count = config["animal"]["env"]["count"]
        self.animals = [Animal(config=config) for _ in range(self.animal_count)]
        self.movement_dim = config["animal"]["init"]["movement_dim"]

        self.drones = []
        for drone_type in self.config.drone:
            count = config["drone"][drone_type]["count"]
            for _ in range(count):
                self.drones.append(Drone(config=config, d_type=drone_type))

        self.drone_count = len(self.drones)

        self.state_counts = {
            "calm": 0,
            "avoid": 0,
            "flee": 0,
        }

        self.reward_stats = {
            "r_monitoring": 0.0,
            "p_disturbance": 0.0,
            "r_bucket": 0.0,
        }

        self.disturbance_sum = 0.0

        self.scale_factor = sum(
            drone_cfg.count * drone_cfg.disturbance_mult
            for drone_cfg in self.config.drone.values()
        )

        self._env_steps = 0
        self.episode = -1

        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.enable_step_logging = False
        self.last_step_stats = {}

        self.out_of_view_steps = 0

    # -------------------------------------------------------------------------
    # config helpers
    # -------------------------------------------------------------------------
    def _cfg(self, key, default):
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict) and key in self.config:
            return self.config[key]
        try:
            return self.config[key]
        except Exception:
            return default

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def _create_resource_map(self):
        if type(self.config["animal"]["init"]["behavior"]) == CRW_CFG:
            return None
        return ResourceMap(config=self.config, seed=self.resource_map_seed)

    def _init_animal(self):
        for animal in self.animals:
            animal.disturbance = 0.0
            animal.escape_dir = np.zeros(3, dtype=np.float32)
            animal.resource_map = self.resource_map

            if getattr(animal.behavior, "handles_spawn", False):
                animal.behavior.reset(animal=animal, rng=self.env_rng)
            else:
                animal.vel_dir = Vector().random_unit(dim=self.movement_dim, rng=self.env_rng)
                animal.vel_speed = self.env_rng.uniform(animal.min_speed, animal.max_speed)

                spawn_dir = Vector().random_unit(dim=self.movement_dim, rng=self.env_rng)
                radius = self.env_rng.uniform(0, self.spawn_radius)
                animal.pos = spawn_dir.scale(radius)
                animal.behavior.reset()

    def _init_drone(self):
        for i in range(self.drone_count):
            animal = self.animals[i % self.animal_count]
            drone = self.drones[i]
            drone.target_id = animal.id

            animal_pos = animal.pos.getter()

            drone.view_dir.unit()
            drone.view_dir.rotate_z(self.env_rng.uniform(-20.0, 20.0))

            view_dir = drone.view_dir.getter()
            inv_scale_view_dir = view_dir.scale(-self.env_rng.uniform(*drone.spawn_dist))
            new_drone_pos = animal_pos.add(inv_scale_view_dir)

            drone.pos.setter(new_drone_pos)
            drone.vel_speed = self.env_rng.uniform(drone.min_speed, min(5.0, drone.max_speed))

    def sample_action(self):
        actions = []

        for drone in self.drones:
            vel_dir = Vector().random_unit(dim=MovementDim.THREE_D, rng=self.sample_action_rng)
            vx, vy, vz = vel_dir.to_numpy()

            vel_speed = self.sample_action_rng.uniform(drone.min_speed, drone.max_speed)
            theta = self.sample_action_rng.triangular(
                -drone.max_cam_rot,
                0,
                drone.max_cam_rot,
            )

            actions.extend([vx, vy, vz, vel_speed, theta])

        return np.array(actions, dtype=np.float32)

    def set_seed(self, seed):
        if seed is None:
            self.curr_episode_seed = self.next_episode_seed
        else:
            self.curr_episode_seed = seed

        self.seeds = np.random.SeedSequence(self.curr_episode_seed).spawn(2)
        self.env_rng = np.random.default_rng(self.seeds[0])
        self.sample_action_rng = np.random.default_rng(self.seeds[1])

        self.next_episode_seed = self.env_rng.integers(0, np.iinfo(np.int32).max)
        self.resource_map_seed = int(self.env_rng.integers(0, np.iinfo(np.int32).max))

    def reset(self, seed=None):
        self._env_steps = 0
        self.lost_track_counter = 0
        self.episode += 1
        self.out_of_view_steps = 0

        self.state_counts = {"calm": 0, "avoid": 0, "flee": 0}
        self.reward_stats = {
            "r_monitoring": 0.0,
            "p_disturbance": 0.0,
            "r_bucket": 0.0,
            "episode_progress": 0.0,
        }

        if self.enable_step_logging:
            self.last_step_stats = {
                "reward": 0.0,
                "monitor_reward": 0.0,
                "disturbance_penalty": 0.0,
                "calm_frac": 0.0,
                "avoid_frac": 0.0,
                "flee_frac": 0.0,
                "mean_disturbance": 0.0,
                "r_bucket": 0.0,
            }

        self.disturbance_sum = 0.0
        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.set_seed(seed)
        self.resource_map = self._create_resource_map()

        self._init_animal()
        self._init_drone()

        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        info = {}
        return observations, info

    def reset_episode_id(self):
        self.episode = -1

    # -------------------------------------------------------------------------
    # actions
    # -------------------------------------------------------------------------
    def package_actions(self, actions):
        match self.config.model.space.action_type:
            case "rel":
                return self.rel_package_actions(actions)
            case "abs":
                return self.abs_package_actions(actions)
            case _:
                raise NotImplementedError(
                    f"action type not implemented: {self.config.model.space.action_type}"
                )

    def abs_package_actions(self, actions):
        packaged_actions = []

        for i, drone_actions in enumerate(actions):
            drone = self.drones[i]

            min_speed = drone.min_speed
            max_speed = drone.max_speed
            max_cam_rot = drone.max_cam_rot

            norm_speed = drone_actions[3]
            norm_theta = drone_actions[4]

            packaged_actions.append(
                {
                    "vel_dir": Vector(drone_actions[0], drone_actions[1], drone_actions[2]),
                    "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                    "theta": norm_theta * max_cam_rot,
                }
            )

        return packaged_actions

    def rel_package_actions(self, actions):
        packaged_actions = []

        for i, drone_actions in enumerate(actions):
            drone = self.drones[i]

            min_speed = drone.min_speed
            max_speed = drone.max_speed
            max_cam_rot = drone.max_cam_rot

            v_forward, v_right, v_up = drone_actions[:3]
            norm_speed = drone_actions[3]
            norm_theta = drone_actions[4]

            x, y, z = self._camera_basis(drone)
            world_vec = v_forward * x + v_right * y + v_up * z

            packaged_actions.append(
                {
                    "vel_dir": Vector(*world_vec),
                    "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                    "theta": norm_theta * max_cam_rot,
                }
            )

        return packaged_actions

    # -------------------------------------------------------------------------
    # stepping
    # -------------------------------------------------------------------------
    def _step_drone(self, drones, actions):
        cam_interp_alpha = 0.25

        for drone, action in zip(drones, actions):
            drone.vel_dir.setter(action["vel_dir"])
            drone.vel_dir.unit()

            drone.vel_speed = action["vel_speed"]
            drone.enforce_speed()

            drone.update_pos()
            drone.enforce_position()

            prev_view = drone.view_dir.to_numpy().astype(np.float32).copy()
            prev_yaw = np.arctan2(prev_view[1], prev_view[0])
            target_yaw = prev_yaw + np.deg2rad(action["theta"])
            new_yaw = (1.0 - cam_interp_alpha) * prev_yaw + cam_interp_alpha * target_yaw

            xy_norm = np.linalg.norm(prev_view[:2])
            if xy_norm < 1e-8:
                xy_norm = 1.0

            new_view = np.array(
                [
                    xy_norm * np.cos(new_yaw),
                    xy_norm * np.sin(new_yaw),
                    prev_view[2],
                ],
                dtype=np.float32,
            )

            new_view = new_view / (np.linalg.norm(new_view) + 1e-8)
            drone.view_dir.setter(Vector(*new_view))
            drone.theta = action["theta"]

    def _step_animal(self):
        segment_complete = False

        for animal in self.animals:
            if animal.behavior.can_flee:
                D = animal.disturbance

                if D > 0.70:
                    state = "flee"
                    animal.vel_dir.setter(Vector(*animal.escape_dir))
                    animal.vel_speed = animal.max_speed

                elif D > 0.50:
                    state = "avoid"
                    animal.update_vel(rng=self.env_rng)

                    base = animal.vel_dir.to_numpy()
                    flee = animal.escape_dir
                    blended = 0.5 * base + 0.5 * flee
                    animal.vel_dir.setter(Vector(*blended))
                else:
                    state = "calm"
                    animal.update_vel(rng=self.env_rng)

                animal.enforce_speed()
            else:
                D = animal.disturbance
                state = "calm"
                a_segment_complete = animal.update_vel(rng=self.env_rng)
                if not segment_complete and a_segment_complete:
                    segment_complete = True

            animal.state = state
            self.state_counts[state] += 1
            self.disturbance_sum += D

            animal.update_pos()
            animal.enforce_position()

        return segment_complete

    # -------------------------------------------------------------------------
    # stats
    # -------------------------------------------------------------------------
    def get_behavior_stats(self):
        if self._env_steps == 0:
            return None

        return {
            "calm_frac": self.state_counts["calm"] / self._env_steps,
            "avoid_frac": self.state_counts["avoid"] / self._env_steps,
            "flee_frac": self.state_counts["flee"] / self._env_steps,
        }

    def get_reward_stats(self):
        if self._env_steps == 0:
            return None

        return {
            "r_monitoring": self.reward_stats["r_monitoring"] / self._env_steps,
            "p_disturbance": self.reward_stats["p_disturbance"] / self._env_steps,
            "r_bucket": self.reward_stats["r_bucket"] / self._env_steps,
            "episode_progress": self._env_steps / self.config.max_episode_steps,
        }

    def set_render_mode(self, mode):
        self.render_mode = mode

    def render(self, fov=None, reward=None):
        self.viewer.draw(
            self.drones,
            self.animals,
            self.render_mode,
            fov=fov,
            reward=reward,
        )

    def torch_to_vec(self, actions):
        actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)

        formatted = []
        for a in actions:
            formatted.append(
                {
                    "vel_dir": Vector(a[0], a[1], a[2]),
                    "vel_speed": float(a[3]),
                    "theta": float(a[4]),
                }
            )
        return formatted

    # -------------------------------------------------------------------------
    # geometry / observations
    # -------------------------------------------------------------------------
    def _camera_basis(self, drone):
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        x = drone.view_dir.to_numpy()
        x = x / (np.linalg.norm(x) + 1e-8)

        y = np.cross(world_z, x)
        y = y / (np.linalg.norm(y) + 1e-8)

        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-8)

        return x, y, z

    def _compute_geometry(self):
        geometry = []

        drone_positions = [d.pos.to_numpy() for d in self.drones]
        animal_positions = [a.pos.to_numpy() for a in self.animals]

        for drone_pos in drone_positions:
            drone_geom = []

            for animal_pos in animal_positions:
                rel_vec = drone_pos - animal_pos
                distance = np.linalg.norm(rel_vec)

                if distance > 1e-8:
                    dir_unit = rel_vec / distance
                else:
                    dir_unit = np.zeros(3, dtype=np.float32)

                drone_geom.append(
                    {
                        "rel_vec": rel_vec,
                        "distance": distance,
                        "dir_unit": dir_unit,
                    }
                )

            geometry.append(drone_geom)

        return geometry

    def _relative_view_angle_bucket(self, animal, drone):
        animal_heading = animal.vel_dir.to_numpy().astype(np.float32).copy()
        animal_to_drone = (drone.pos.to_numpy() - animal.pos.to_numpy()).astype(np.float32)

        animal_heading[2] = 0.0
        animal_to_drone[2] = 0.0

        h_norm = np.linalg.norm(animal_heading[:2])
        r_norm = np.linalg.norm(animal_to_drone[:2])

        if h_norm < 1e-8 or r_norm < 1e-8:
            return None, None

        h = animal_heading[:2] / h_norm
        r = animal_to_drone[:2] / r_norm

        det = h[0] * r[1] - h[1] * r[0]
        dot = np.clip(np.dot(h, r), -1.0, 1.0)
        angle_deg = np.degrees(np.arctan2(det, dot)) % 360.0

        if angle_deg >= 315.0 or angle_deg < 45.0:
            bucket = 0  # front
        elif angle_deg < 135.0:
            bucket = 1  # left
        elif angle_deg < 225.0:
            bucket = 2  # back
        else:
            bucket = 3  # right

        return angle_deg, bucket

    def _update_view_buckets(self, observations):
        drone_feat_dim = self.config.model.space.drone_features
        animal_feat_dim = self.config.model.space.animal_features
        animal_obs = observations[
            :,
            drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim,
        ].reshape(self.drone_count, self.animal_count, animal_feat_dim)

        for d, drone in enumerate(self.drones):
            for a, animal in enumerate(self.animals):
                in_view = animal_obs[d, a, 0] == 1.0
                if not in_view:
                    continue

                _, bucket = self._relative_view_angle_bucket(animal, drone)
                if bucket is None:
                    continue

                self.view_bucket_counts[a, bucket] += 1.0
                self.view_bucket_totals[a] += 1.0

    def _build_observations(self, geometry):
        obs_all = []
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for d, drone in enumerate(self.drones):
            x = drone.view_dir.to_numpy()
            x = x / (np.linalg.norm(x) + 1e-8)

            altitude = drone.pos.to_numpy()[2]
            altitude_norm = altitude / (drone.max_altitude + 1e-8)

            drone_features = [x[0], x[1], x[2], altitude_norm]

            y = np.cross(world_z, x)
            y = y / (np.linalg.norm(y) + 1e-8)
            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-8)

            v_max = np.deg2rad(drone.ver_angle / 2)
            h_max = np.deg2rad(drone.hor_angle / 2)

            target_idx = d % self.animal_count

            animal_features = []
            for a, animal in enumerate(self.animals):
                rel_unit = -geometry[d][a]["dir_unit"]
                distance = geometry[d][a]["distance"]

                cx = np.dot(rel_unit, x)
                cy = np.dot(rel_unit, y)
                cz = np.dot(rel_unit, z)

                v_angle = np.arctan2(cz, cx)
                h_angle = np.arctan2(cy, cx)

                in_view = (
                    cx > 0
                    and abs(v_angle) <= v_max
                    and abs(h_angle) <= h_max
                    and distance <= drone.view_range
                )

                animal_vel = (
                    animal.vel_dir.to_numpy()
                    * (animal.vel_speed / (animal.max_speed + 1e-8))
                )

                v_cam_x = np.dot(animal_vel, x)
                v_cam_y = np.dot(animal_vel, y)
                v_cam_z = np.dot(animal_vel, z)

                if in_view:
                    animal_features.extend(
                        [
                            1.0,
                            distance / drone.view_range,
                            v_angle / (v_max + 1e-8),
                            h_angle / (h_max + 1e-8),
                            v_cam_x,
                            v_cam_y,
                            v_cam_z,
                            float(a == target_idx),
                        ]
                    )
                else:
                    animal_features.extend(
                        [
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )

            obs_all.append(np.array(drone_features + animal_features, dtype=np.float32))

        return np.array(obs_all, dtype=np.float32)

    # -------------------------------------------------------------------------
    # disturbance
    # -------------------------------------------------------------------------
    def _compute_disturbance(self, geometry):
        for a, animal in enumerate(self.animals):
            escape_vec = np.zeros(3, dtype=np.float32)
            animal.disturbance = 0.0
            disturbances = []
            escape_vecs = []

            for d, drone in enumerate(self.drones):
                rel_vec = geometry[d][a]["rel_vec"]
                distance = geometry[d][a]["distance"]

                if distance > 1e-8:
                    unit_escape_vec = -rel_vec / distance
                else:
                    unit_escape_vec = np.zeros(3, dtype=np.float32)

                escape_vecs.append(unit_escape_vec)

                gain = disturbance_gain(
                    rel_vec,
                    drone.vel_dir.to_numpy(),
                    animal.vel_dir.to_numpy(),
                    self.config,
                ) * drone.disturbance_mult

                disturbances.append(gain)

            sorted_idx = np.argsort(disturbances)[::-1]
            for i in range(len(disturbances)):
                if animal.disturbance >= 1.0:
                    break

                idx = sorted_idx[i]
                disturbance = disturbances[idx]
                influence = (1.0 - animal.disturbance) * disturbance
                animal.disturbance += influence
                escape_vec += escape_vecs[idx] * influence

            norm = np.linalg.norm(escape_vec)
            if norm > 1e-8:
                animal.escape_dir = escape_vec / norm
            else:
                animal.escape_dir = animal.vel_dir.to_numpy()

            animal.disturbance = float(np.clip(animal.disturbance, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # reward helpers
    # -------------------------------------------------------------------------
    def _bucket_balance_score(self):
        min_views_before_balance = 8
        scores = []

        for a in range(self.animal_count):
            total = self.view_bucket_totals[a]
            if total < min_views_before_balance:
                continue

            p = self.view_bucket_counts[a] / (total + 1e-8)
            score = 1.0 - (np.std(p) / 0.4330127)
            score = np.clip(score, 0.0, 1.0)
            scores.append(score)

        if len(scores) == 0:
            return 0.0

        return float(np.mean(scores))

    # -------------------------------------------------------------------------
    # reward
    # -------------------------------------------------------------------------
    def compute_reward(self, observations, actions):
        pure_monitor_terms = []
        interpolated_terms = []

        for d in range(self.drone_count):
            drone_obs = observations[d]

            drone_feat_dim = self.config.model.space.drone_features
            animal_feat_dim = self.config.model.space.animal_features
            animal_obs = drone_obs[
                drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim
            ].reshape(self.animal_count, animal_feat_dim)

            target = animal_obs[:, 7]
            is_target = target > 0.5

            if not np.any(is_target):
                continue

            idx = int(np.where(is_target)[0][0])

            rel_vec = self.drones[d].pos.to_numpy() - self.animals[idx].pos.to_numpy()
            x, z = rel_vec_to_xz(rel_vec)

            mon = float(disturbance_monitor_reward(x, z))
            raw_dist = float(raw_disturbance_calc(x, z))

            interp = float(
                interpolate_monitor_disturbance(
                    self.config.alpha,
                    mon,
                    raw_dist,
                    x,
                    z,
                )
            )
            interp = float(np.clip(interp, 0.0, 1.0))

            pure_monitor_terms.append(mon)
            interpolated_terms.append(interp)

        monitor_reward = float(np.mean(pure_monitor_terms)) if pure_monitor_terms else 0.0
        disturbance_penalty = float(np.mean([a.disturbance for a in self.animals]))
        r_bucket = self._bucket_balance_score()

        base_reward = float(np.mean(interpolated_terms)) if interpolated_terms else 0.0
        bucket_scale = float(self._cfg("bucket_reward_scale", 0.15))

        final_reward = base_reward + bucket_scale * r_bucket
        final_reward = float(np.clip(final_reward, 0.0, 1.0))

        self.reward_stats["r_monitoring"] += monitor_reward
        self.reward_stats["p_disturbance"] += disturbance_penalty
        self.reward_stats["r_bucket"] += r_bucket

        if self.enable_step_logging:
            behavior_counts = {"calm": 0, "avoid": 0, "flee": 0}
            for animal in self.animals:
                behavior_counts[animal.state] += 1

            n = max(self.animal_count, 1)
            self.last_step_stats = {
                "reward": float(final_reward),
                "monitor_reward": float(monitor_reward),
                "disturbance_penalty": float(disturbance_penalty),
                "calm_frac": behavior_counts["calm"] / n,
                "avoid_frac": behavior_counts["avoid"] / n,
                "flee_frac": behavior_counts["flee"] / n,
                "mean_disturbance": float(np.mean([a.disturbance for a in self.animals])),
                "r_bucket": float(r_bucket),
            }

        return float(final_reward), monitor_reward, disturbance_penalty

    # -------------------------------------------------------------------------
    # termination
    # -------------------------------------------------------------------------
    def _check_termination(self, observations):
        if self._env_steps >= self.config["max_episode_steps"]:
            return True

        drone_feat_dim = self.config.model.space.drone_features
        animal_feat_dim = self.config.model.space.animal_features
        animal_obs = observations[
            :,
            drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim,
        ].reshape(self.drone_count, self.animal_count, animal_feat_dim)

        visible = animal_obs[:, :, 0] > 0.5
        target_visible = np.any(visible, axis=0)
        target_visible_count = np.sum(target_visible)

        if target_visible_count <= int(self.animal_count * 0.5):
            self.lost_track_counter += 1
            if self.lost_track_counter >= self.config.track_loss_grace:
                return True
        else:
            self.lost_track_counter = 0

        return False

    # -------------------------------------------------------------------------
    # main step
    # -------------------------------------------------------------------------
    def step(self, actions):
        self._env_steps += 1

        actions = self.package_actions(actions)
        self._step_drone(self.drones, actions)

        geometry = self._compute_geometry()
        self._compute_disturbance(geometry)
        segment_complete = self._step_animal()

        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        self._update_view_buckets(observations)

        reward, monitor_r, disturbance_p = self.compute_reward(observations, actions)

        terminated = segment_complete or self._check_termination(observations)
        truncated = False

        info = {
            "fov": observations,
            "reward": float(reward),
            "monitor_reward": float(monitor_r),
            "disturbance_penalty": float(disturbance_p),
        }

        if self.render_mode is not None:
            self.render(fov=observations, reward=reward)

        return observations, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # logging
    # -------------------------------------------------------------------------
    def step_log(self):
        rows = []

        for a_idx, animal in enumerate(self.animals):
            rows.append(
                {
                    "episode": self.episode,
                    "step": self._env_steps,
                    "entity_type": "animal",
                    "id": animal.id,
                    "x": animal.pos.x,
                    "y": animal.pos.y,
                    "z": animal.pos.z,
                    "vx": animal.vel_dir.x * animal.vel_speed,
                    "vy": animal.vel_dir.y * animal.vel_speed,
                    "vz": animal.vel_dir.z * animal.vel_speed,
                    "speed": animal.vel_speed,
                    "view_x": "",
                    "view_y": "",
                    "view_z": "",
                    "state": animal.state,
                    "disturbance": animal.disturbance,
                    "bucket_front": self.view_bucket_counts[a_idx, 0],
                    "bucket_left": self.view_bucket_counts[a_idx, 1],
                    "bucket_back": self.view_bucket_counts[a_idx, 2],
                    "bucket_right": self.view_bucket_counts[a_idx, 3],
                }
            )

        for drone in self.drones:
            rows.append(
                {
                    "episode": self.episode,
                    "step": self._env_steps,
                    "entity_type": drone.drone_type,
                    "id": drone.id,
                    "x": drone.pos.x,
                    "y": drone.pos.y,
                    "z": drone.pos.z,
                    "vx": drone.vel_dir.x * drone.vel_speed,
                    "vy": drone.vel_dir.y * drone.vel_speed,
                    "vz": drone.vel_dir.z * drone.vel_speed,
                    "speed": drone.vel_speed,
                    "view_x": drone.view_dir.x,
                    "view_y": drone.view_dir.y,
                    "view_z": drone.view_dir.z,
                    "state": "",
                    "disturbance": "",
                    "bucket_front": "",
                    "bucket_left": "",
                    "bucket_back": "",
                    "bucket_right": "",
                    **self.last_step_stats,
                }
            )

        return rows