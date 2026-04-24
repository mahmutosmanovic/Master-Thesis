import numpy as np
from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .disturbance import disturbance_gain, analytic_upper_bound
from .resource_map import ResourceMap
from .immutables import MovementDim
from .behaviors import CRW_CFG


class Env:
    def __init__(self, config, render_mode=None, seed=42):
        self.config = config
        self.render_mode = render_mode
        self.alpha = config.alpha

        self.viewer = Viewer(config)
        self.resource_map = None
        self.lost_track_counter = 0
        self.out_of_view_steps = 0

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
        self.max_view_range = max(d.view_range for d in self.drones)

        self._init_space_config()
        self._init_safety_config()
        self._init_orbit_config()
        self._init_view_sep_config()
        self._init_monitoring_config()

        self.state_counts = {"calm": 0, "avoid": 0, "flee": 0}
        self.reward_stats = self._empty_reward_stats()

        self.disturbance_sum = 0.0
        self._env_steps = 0
        self.episode = -1

        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.prev_vel_dirs = None
        self.prev_orbit_signs = np.zeros(self.drone_count, dtype=np.float32)

        self.enable_step_logging = False
        self.last_step_stats = {}

        self.reward_scales = self._compute_reward_scales()

        self.last_hard_safety_violation = False
        self.last_min_drone_drone_dist = np.inf
        self.last_min_drone_animal_dist = np.inf

        self.last_max_single_disturbance = 0.0
        self.last_target_visible_fraction = 0.0

        self.set_seed(seed)

    def _compute_reward_scales(self):
        if len(self.drones) == 0:
            return np.ones(0, dtype=np.float32)

        reward_scales = []
        for drone in self.drones:
            base_max, _, _ = analytic_upper_bound(
                self.config.XY_scale,
                self.config.Z_scale,
                drone.view_range,
                self.alpha,
            )
            reward_scales.append(np.clip(base_max, 1e-6, 1.0))

        return np.array(reward_scales, dtype=np.float32)

    def _cfg_get(self, *keys, default=None):
        obj = self.config
        for key in keys:
            if isinstance(obj, dict):
                if key not in obj:
                    return default
                obj = obj[key]
            else:
                if not hasattr(obj, key):
                    return default
                obj = getattr(obj, key)
        return obj

    def _init_space_config(self):
        self.drone_feat_dim = int(self._cfg_get("model", "space", "drone_features", default=8))
        self.other_drone_feat_dim = int(self._cfg_get("model", "space", "other_drone_features", default=4))
        self.animal_feat_dim = int(self._cfg_get("model", "space", "animal_features", default=8))

    def _init_safety_config(self):
        self.drone_proximity_range = float(self._cfg_get("drone_proximity_range", default=50.0))
        self.drone_proximity_scale = float(self._cfg_get("drone_proximity_scale", default=0.75))
        self.drone_proximity_exp = float(self._cfg_get("drone_proximity_exp", default=1.0))

        self.animal_proximity_range = float(self._cfg_get("animal_proximity_range", default=15.0))
        self.animal_proximity_scale = float(self._cfg_get("animal_proximity_scale", default=0.30))
        self.animal_proximity_exp = float(self._cfg_get("animal_proximity_exp", default=1.5))

        self.hard_safety_radius = float(self._cfg_get("hard_safety_radius", default=5.0))
        self.hard_safety_penalty = float(self._cfg_get("hard_safety_penalty", default=1.5))

        self.reset_safety_margin = float(self._cfg_get("reset_safety_margin", default=20.0))
        self.max_reset_tries = int(self._cfg_get("max_reset_tries", default=100))

    def _init_orbit_config(self):
        self.p_orbit_tangent_scale = float(self._cfg_get("p_orbit_tangent_scale", default=0.10))
        self.p_orbit_persistence_scale = float(self._cfg_get("p_orbit_persistence_scale", default=0.05))
        self.orbit_sign_deadzone = float(self._cfg_get("orbit_sign_deadzone", default=0.03))

    def _init_view_sep_config(self):
        self.view_sep_weight = float(self._cfg_get("view_sep_weight", default=0.10))
        self.min_view_sep_distance = float(self._cfg_get("min_view_sep_distance", default=5.0))
        self.require_target_visible_for_sep = bool(self._cfg_get("require_target_visible_for_sep", default=False))

    def _init_monitoring_config(self):
        self.reaction_threshold_single = float(self._cfg_get("reaction_threshold_single", default=0.40))
        self.flee_threshold_total = float(self._cfg_get("flee_threshold_total", default=0.70))
        self.not_visible_penalty = float(self._cfg_get("not_visible_penalty", default=1.25))
        self.not_visible_penalty_power = float(self._cfg_get("not_visible_penalty_power", default=2.0))
        self.all_view_bonus = float(self._cfg_get("all_view_bonus", default=0.10))

    def _empty_reward_stats(self):
        return {
            "r_monitoring": 0.0,
            "p_disturbance": 0.0,
            "r_vis": 0.0,
            "r_dist": 0.0,
            "r_align": 0.0,
            "r_bucket": 0.0,
            "r_view_sep": 0.0,
            "r_all_view": 0.0,
            "p_not_visible": 0.0,
            "p_drone_proximity": 0.0,
            "p_animal_proximity": 0.0,
            "p_hard_safety": 0.0,
            "p_orbit_tangent": 0.0,
            "p_orbit_persistence": 0.0,
            "episode_progress": 0.0,
        }

    def _obs_offsets(self):
        other_block = (self.drone_count - 1) * self.other_drone_feat_dim
        animal_block = self.animal_count * self.animal_feat_dim
        other_start = self.drone_feat_dim
        animal_start = other_start + other_block
        total = self.drone_feat_dim + other_block + animal_block
        return other_start, animal_start, total

    def _extract_animal_obs(self, observations):
        _, animal_start, _ = self._obs_offsets()
        animal_flat = observations[:, animal_start:animal_start + self.animal_count * self.animal_feat_dim]
        return animal_flat.reshape(self.drone_count, self.animal_count, self.animal_feat_dim)

    def _create_resource_map(self):
        if type(self.config["animal"]["init"]["behavior"]) == CRW_CFG:
            return None
        return ResourceMap(config=self.config, seed=self.resource_map_seed)

    def _init_animal(self):
        for animal in self.animals:
            animal.disturbance = 0.0
            animal.max_single_disturbance = 0.0
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
        for i, drone in enumerate(self.drones):
            animal = self.animals[i % self.animal_count]
            drone.target_id = animal.id

            animal_pos = animal.pos.getter()

            drone.view_dir.unit()
            drone.view_dir.rotate_z(self.env_rng.uniform(-20.0, 20.0))

            view_dir = drone.view_dir.getter()
            inv_scale_view_dir = view_dir.scale(-self.env_rng.uniform(*drone.spawn_dist))
            new_drone_pos = animal_pos.add(inv_scale_view_dir)

            drone.pos.setter(new_drone_pos)
            drone.vel_speed = self.env_rng.uniform(drone.min_speed, min(5.0, drone.max_speed))

    def _is_safe_initial_configuration(self):
        drone_positions = [d.pos.to_numpy().astype(np.float32) for d in self.drones]
        animal_positions = [a.pos.to_numpy().astype(np.float32) for a in self.animals]
        min_allowed = self.hard_safety_radius + self.reset_safety_margin

        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                if np.linalg.norm(drone_positions[i] - drone_positions[j]) < min_allowed:
                    return False

        for dp in drone_positions:
            for ap in animal_positions:
                if np.linalg.norm(dp - ap) < min_allowed:
                    return False

        return True

    def sample_action(self):
        actions = []
        for drone in self.drones:
            vel_dir = Vector().random_unit(dim=MovementDim.THREE_D, rng=self.sample_action_rng)
            vx, vy, vz = vel_dir.to_numpy()

            vel_speed = self.sample_action_rng.uniform(drone.min_speed, drone.max_speed)
            theta = self.sample_action_rng.triangular(-drone.max_cam_rot, 0, drone.max_cam_rot)

            actions.extend([vx, vy, vz, vel_speed, theta])

        return np.array(actions, dtype=np.float32)

    def set_seed(self, seed):
        if hasattr(self, "next_episode_seed") and seed is None:
            self.curr_episode_seed = self.next_episode_seed
        else:
            self.curr_episode_seed = 42 if seed is None else seed

        self.seeds = np.random.SeedSequence(self.curr_episode_seed).spawn(2)
        self.env_rng = np.random.default_rng(self.seeds[0])
        self.sample_action_rng = np.random.default_rng(self.seeds[1])

        self.next_episode_seed = int(self.env_rng.integers(0, np.iinfo(np.int32).max))
        self.resource_map_seed = int(self.env_rng.integers(0, np.iinfo(np.int32).max))

    def reset(self, seed=None):
        self._env_steps = 0
        self.lost_track_counter = 0
        self.out_of_view_steps = 0
        self.episode += 1

        self.state_counts = {"calm": 0, "avoid": 0, "flee": 0}
        self.reward_stats = self._empty_reward_stats()
        self.disturbance_sum = 0.0

        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.prev_orbit_signs = np.zeros(self.drone_count, dtype=np.float32)

        self.last_hard_safety_violation = False
        self.last_min_drone_drone_dist = np.inf
        self.last_min_drone_animal_dist = np.inf
        self.last_max_single_disturbance = 0.0
        self.last_target_visible_fraction = 0.0

        if self.enable_step_logging:
            self.last_step_stats = {
                "reward": 0.0,
                "monitor_reward": 0.0,
                "disturbance_penalty": 0.0,
                "max_single_disturbance": 0.0,
                "target_visible_fraction": 0.0,
                "calm_frac": 0.0,
                "avoid_frac": 0.0,
                "flee_frac": 0.0,
                "mean_disturbance": 0.0,
                "r_vis": 0.0,
                "r_dist": 0.0,
                "r_align": 0.0,
                "r_view_sep": 0.0,
                "r_all_view": 0.0,
                "p_not_visible": 0.0,
                "p_drone_proximity": 0.0,
                "p_animal_proximity": 0.0,
                "p_hard_safety": 0.0,
                "p_orbit_tangent": 0.0,
                "p_orbit_persistence": 0.0,
                "hard_safety_violation": 0.0,
                "min_drone_drone_dist": np.inf,
                "min_drone_animal_dist": np.inf,
            }

        self.set_seed(seed)
        self.resource_map = self._create_resource_map()

        success = False
        for _ in range(self.max_reset_tries):
            self._init_animal()
            self._init_drone()
            if self._is_safe_initial_configuration():
                success = True
                break

        if not success:
            raise RuntimeError("Could not sample a safe initial configuration.")

        self.prev_vel_dirs = [
            drone.vel_dir.to_numpy().astype(np.float32).copy()
            for drone in self.drones
        ]

        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        self.reward_scales = self._compute_reward_scales()

        return observations, {}

    def reset_episode_id(self):
        self.episode = -1

    def package_actions(self, actions):
        match self.config.model.space.action_type:
            case "rel":
                return self.rel_package_actions(actions)
            case "abs":
                return self.abs_package_actions(actions)
            case _:
                raise NotImplementedError(f"Unknown action type: {self.config.model.space.action_type}")

    def abs_package_actions(self, actions):
        packaged_actions = []

        for i, drone_actions in enumerate(actions):
            drone = self.drones[i]
            min_speed = drone.min_speed
            max_speed = drone.max_speed
            max_cam_rot = drone.max_cam_rot

            norm_speed = drone_actions[3]
            norm_theta = drone_actions[4]

            packaged_actions.append({
                "vel_dir": Vector(drone_actions[0], drone_actions[1], drone_actions[2]),
                "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                "theta": norm_theta * max_cam_rot,
            })

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

            packaged_actions.append({
                "vel_dir": Vector(*world_vec),
                "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                "theta": norm_theta * max_cam_rot,
            })

        return packaged_actions

    def _step_drone(self, drones, actions):
        cam_interp_alpha = 0.25

        for drone, action in zip(drones, actions):
            drone.vel_dir.setter(action["vel_dir"])
            drone.vel_dir.unit()

            drone.vel_speed = action["vel_speed"]
            drone.enforce_speed()

            drone.update_pos()
            drone.enforce_position()

            prev_view = drone.view_dir.to_numpy().astype(np.float32)
            prev_yaw = np.arctan2(prev_view[1], prev_view[0])
            target_yaw = prev_yaw + np.deg2rad(action["theta"])
            new_yaw = (1.0 - cam_interp_alpha) * prev_yaw + cam_interp_alpha * target_yaw

            xy_norm = np.linalg.norm(prev_view[:2])
            if xy_norm < 1e-8:
                xy_norm = 1.0

            new_view = np.array([
                xy_norm * np.cos(new_yaw),
                xy_norm * np.sin(new_yaw),
                prev_view[2],
            ], dtype=np.float32)
            new_view /= (np.linalg.norm(new_view) + 1e-8)

            drone.view_dir.setter(Vector(*new_view))
            drone.theta = action["theta"]

    def _step_animal(self):
        segment_complete = False

        for animal in self.animals:
            D_total = animal.disturbance
            D_single = animal.max_single_disturbance
            reaction_level = max(D_total, D_single)

            if animal.behavior.can_flee:
                if D_total > self.flee_threshold_total:
                    state = "flee"
                    animal.vel_dir.setter(Vector(*animal.escape_dir))
                    animal.vel_speed = animal.max_speed
                elif reaction_level > self.reaction_threshold_single:
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
                state = "calm"
                a_segment_complete = animal.update_vel(rng=self.env_rng)
                if not segment_complete and a_segment_complete:
                    segment_complete = True

            animal.state = state
            self.state_counts[state] += 1
            self.disturbance_sum += D_total

            animal.update_pos()
            animal.enforce_position()

        return segment_complete

    def get_behavior_stats(self):
        if self._env_steps == 0:
            return None

        n = max(self.animal_count * self._env_steps, 1)
        return {
            "calm_frac": self.state_counts["calm"] / n,
            "avoid_frac": self.state_counts["avoid"] / n,
            "flee_frac": self.state_counts["flee"] / n,
        }

    def get_reward_stats(self):
        if self._env_steps == 0:
            return None

        return {
            "r_monitoring": self.reward_stats["r_monitoring"] / self._env_steps,
            "p_disturbance": self.reward_stats["p_disturbance"] / self._env_steps,
            "r_vis": self.reward_stats["r_vis"] / self._env_steps,
            "r_dist": self.reward_stats["r_dist"] / self._env_steps,
            "r_align": self.reward_stats["r_align"] / self._env_steps,
            "r_bucket": self.reward_stats["r_bucket"] / self._env_steps,
            "r_view_sep": self.reward_stats["r_view_sep"] / self._env_steps,
            "r_all_view": self.reward_stats["r_all_view"] / self._env_steps,
            "p_not_visible": self.reward_stats["p_not_visible"] / self._env_steps,
            "p_drone_proximity": self.reward_stats["p_drone_proximity"] / self._env_steps,
            "p_animal_proximity": self.reward_stats["p_animal_proximity"] / self._env_steps,
            "p_hard_safety": self.reward_stats["p_hard_safety"] / self._env_steps,
            "p_orbit_tangent": self.reward_stats["p_orbit_tangent"] / self._env_steps,
            "p_orbit_persistence": self.reward_stats["p_orbit_persistence"] / self._env_steps,
            "episode_progress": self._env_steps / self.config.max_episode_steps,
        }

    def set_render_mode(self, mode):
        self.render_mode = mode

    def render(self, fov=None, reward=None):
        self.viewer.draw(self.drones, self.animals, self.render_mode, fov=fov, reward=reward)

    def torch_to_vec(self, actions):
        actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)

        formatted = []
        for a in actions:
            formatted.append({
                "vel_dir": Vector(a[0], a[1], a[2]),
                "vel_speed": float(a[3]),
                "theta": float(a[4]),
            })
        return formatted

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

                drone_geom.append({
                    "rel_vec": rel_vec,
                    "distance": distance,
                    "dir_unit": dir_unit,
                })

            geometry.append(drone_geom)

        return geometry

    def _compute_closest_animal_distances(self):
        if self.drone_count == 0 or self.animal_count == 0:
            return np.zeros(self.drone_count, dtype=np.float32)

        drone_positions = [d.pos.to_numpy().astype(np.float32) for d in self.drones]
        animal_positions = [a.pos.to_numpy().astype(np.float32) for a in self.animals]

        closest_dists = np.full(self.drone_count, np.inf, dtype=np.float32)

        for i, dpos in enumerate(drone_positions):
            for apos in animal_positions:
                dist = float(np.linalg.norm(dpos - apos))
                if dist < closest_dists[i]:
                    closest_dists[i] = dist

        return closest_dists

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
            bucket = 0
        elif angle_deg < 135.0:
            bucket = 1
        elif angle_deg < 225.0:
            bucket = 2
        else:
            bucket = 3

        return angle_deg, bucket

    def _update_view_buckets(self, observations):
        animal_obs = self._extract_animal_obs(observations)

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
            x = drone.view_dir.to_numpy().astype(np.float32)
            x = x / (np.linalg.norm(x) + 1e-8)

            y = np.cross(world_z, x)
            y = y / (np.linalg.norm(y) + 1e-8)

            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-8)

            altitude = drone.pos.to_numpy()[2]
            altitude_norm = altitude / (drone.max_altitude + 1e-8)

            drone_features = [
                x[0],
                x[1],
                x[2],
                altitude_norm,
                self.alpha,
                drone.view_range / (self.max_view_range + 1e-8),
                drone.hor_angle / 180.0,
                drone.ver_angle / 180.0,
            ]

            other_drone_features = []
            if self.other_drone_feat_dim > 0:
                curr_pos = drone.pos.to_numpy().astype(np.float32)

                for j, other in enumerate(self.drones):
                    if j == d:
                        continue

                    other_pos = other.pos.to_numpy().astype(np.float32)
                    rel = other_pos - curr_pos

                    rel_x = np.dot(rel, x) / (self.drone_proximity_range + 1e-8)
                    rel_y = np.dot(rel, y) / (self.drone_proximity_range + 1e-8)
                    rel_z = np.dot(rel, z) / (self.drone_proximity_range + 1e-8)
                    rel_dist = np.linalg.norm(rel) / (self.drone_proximity_range + 1e-8)

                    other_drone_features.extend([rel_x, rel_y, rel_z, rel_dist])

            v_max = np.deg2rad(drone.ver_angle / 2.0)
            h_max = np.deg2rad(drone.hor_angle / 2.0)

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
                    cx > 0.0
                    and abs(v_angle) <= v_max
                    and abs(h_angle) <= h_max
                    and distance <= drone.view_range
                )

                animal_vel = animal.vel_dir.to_numpy() * (
                    animal.vel_speed / (animal.max_speed + 1e-8)
                )

                v_cam_x = np.dot(animal_vel, x)
                v_cam_y = np.dot(animal_vel, y)
                v_cam_z = np.dot(animal_vel, z)

                if in_view:
                    animal_features.extend([
                        1.0,
                        distance / (drone.view_range + 1e-8),
                        v_angle / (v_max + 1e-8),
                        h_angle / (h_max + 1e-8),
                        v_cam_x,
                        v_cam_y,
                        v_cam_z,
                        float(animal.id == drone.target_id),
                    ])
                else:
                    animal_features.extend([
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        float(animal.id == drone.target_id),
                    ])

            obs = np.array(
                drone_features + other_drone_features + animal_features,
                dtype=np.float32,
            )
            obs_all.append(obs)

        return np.array(obs_all, dtype=np.float32)

    def _compute_disturbance(self, geometry):
        max_single_overall = 0.0

        for a, animal in enumerate(self.animals):
            escape_vec = np.zeros(3, dtype=np.float32)
            animal.disturbance = 0.0
            animal.max_single_disturbance = 0.0
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

                gain = float(np.clip(gain, 0.0, 1.0))
                disturbances.append(gain)
                animal.max_single_disturbance = max(animal.max_single_disturbance, gain)

            sorted_idx = np.argsort(disturbances)[::-1]
            for idx in sorted_idx:
                if animal.disturbance >= 1.0:
                    break

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
            max_single_overall = max(max_single_overall, animal.max_single_disturbance)

        self.last_max_single_disturbance = float(max_single_overall)

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

        return 0.0 if len(scores) == 0 else float(np.mean(scores))

    def _view_separation_score(self, observations=None):
        scores = []
        visible_mask = None

        if observations is not None:
            animal_obs = self._extract_animal_obs(observations)
            visible_mask = animal_obs[:, :, 0] > 0.5

        for animal in self.animals:
            dirs = []

            for d, drone in enumerate(self.drones):
                if drone.target_id != animal.id:
                    continue

                if self.require_target_visible_for_sep and visible_mask is not None:
                    if not visible_mask[d, animal.id]:
                        continue

                rel = drone.pos.to_numpy().astype(np.float32) - animal.pos.to_numpy().astype(np.float32)
                rel[2] = 0.0

                norm = np.linalg.norm(rel[:2])
                if norm < max(self.min_view_sep_distance, 1e-8):
                    continue

                dirs.append(rel[:2] / norm)

            if len(dirs) < 2:
                continue

            pair_scores = []
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    dot = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
                    pair_scores.append((1.0 - dot) / 2.0)

            if pair_scores:
                scores.append(np.mean(pair_scores))

        return 0.0 if len(scores) == 0 else float(np.mean(scores))

    def _direction_change_penalty(self, drone_idx, drone_action):
        if self.prev_vel_dirs is None:
            return 0.0

        curr_dir = drone_action["vel_dir"].to_numpy().astype(np.float32)
        curr_norm = np.linalg.norm(curr_dir)
        if curr_norm <= 1e-8:
            return 0.0
        curr_dir /= curr_norm

        prev_dir = self.prev_vel_dirs[drone_idx]
        prev_norm = np.linalg.norm(prev_dir)
        if prev_norm > 1e-8:
            prev_dir = prev_dir / prev_norm
        else:
            prev_dir = curr_dir

        cos_turn = np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0)
        reversal_penalty = max(0.0, -cos_turn)
        speed_frac = self.drones[drone_idx].vel_speed / (self.drones[drone_idx].max_speed + 1e-8)

        return reversal_penalty * speed_frac * self.config.p_dir_change_scale

    def _orbit_penalty(self, drone_idx, drone_action):
        if self.animal_count == 0:
            return 0.0, 0.0

        drone = self.drones[drone_idx]

        target_animal = None
        for animal in self.animals:
            if animal.id == drone.target_id:
                target_animal = animal
                break

        if target_animal is None:
            return 0.0, 0.0

        drone_pos = drone.pos.to_numpy().astype(np.float32)
        animal_pos = target_animal.pos.to_numpy().astype(np.float32)

        rel = drone_pos - animal_pos
        rel[2] = 0.0

        rel_norm = np.linalg.norm(rel[:2])
        if rel_norm < 1e-6:
            return 0.0, 0.0

        r_hat = rel[:2] / rel_norm

        vel = drone_action["vel_dir"].to_numpy().astype(np.float32)
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-8:
            return 0.0, 0.0
        vel = vel / vel_norm

        vel_xy = vel[:2]
        radial = np.dot(vel_xy, r_hat) * r_hat
        tangential = vel_xy - radial
        tangential_mag = float(np.linalg.norm(tangential))

        signed_orbit = float(r_hat[0] * vel_xy[1] - r_hat[1] * vel_xy[0])

        if abs(signed_orbit) < self.orbit_sign_deadzone:
            curr_sign = 0.0
        else:
            curr_sign = float(np.sign(signed_orbit))

        prev_sign = self.prev_orbit_signs[drone_idx]
        persistence_penalty = 0.0
        if prev_sign != 0.0 and curr_sign != 0.0 and prev_sign == curr_sign:
            persistence_penalty = abs(signed_orbit)

        self.prev_orbit_signs[drone_idx] = curr_sign

        return tangential_mag, persistence_penalty

    def _closeness_penalty(self, distance, radius, scale, exponent):
        if radius <= 1e-8 or distance >= radius:
            return 0.0

        closeness = 1.0 - (distance / radius)
        closeness = np.clip(closeness, 0.0, 1.0)

        denom = np.exp(exponent) - 1.0
        if abs(denom) < 1e-8:
            shaped = closeness
        else:
            shaped = (np.exp(exponent * closeness) - 1.0) / denom

        return float(scale * shaped)

    def _compute_proximity_terms(self):
        drone_positions = [d.pos.to_numpy().astype(np.float32) for d in self.drones]
        animal_positions = [a.pos.to_numpy().astype(np.float32) for a in self.animals]

        drone_drone_penalties = []
        drone_animal_penalties = []

        min_drone_drone_dist = np.inf
        min_drone_animal_dist = np.inf
        hard_violation = False

        for i in range(self.drone_count):
            for j in range(i + 1, self.drone_count):
                dist = float(np.linalg.norm(drone_positions[i] - drone_positions[j]))
                min_drone_drone_dist = min(min_drone_drone_dist, dist)

                drone_drone_penalties.append(
                    self._closeness_penalty(
                        distance=dist,
                        radius=self.drone_proximity_range,
                        scale=self.drone_proximity_scale,
                        exponent=self.drone_proximity_exp,
                    )
                )

                if dist < self.hard_safety_radius:
                    hard_violation = True

        for i in range(self.drone_count):
            for a in range(self.animal_count):
                dist = float(np.linalg.norm(drone_positions[i] - animal_positions[a]))
                min_drone_animal_dist = min(min_drone_animal_dist, dist)

                drone_animal_penalties.append(
                    self._closeness_penalty(
                        distance=dist,
                        radius=self.animal_proximity_range,
                        scale=self.animal_proximity_scale,
                        exponent=self.animal_proximity_exp,
                    )
                )

                if dist < self.hard_safety_radius:
                    hard_violation = True

        p_drone_proximity = float(np.mean(drone_drone_penalties)) if drone_drone_penalties else 0.0
        p_animal_proximity = float(np.mean(drone_animal_penalties)) if drone_animal_penalties else 0.0
        p_hard_safety = float(self.hard_safety_penalty if hard_violation else 0.0)

        self.last_hard_safety_violation = hard_violation
        self.last_min_drone_drone_dist = float(min_drone_drone_dist)
        self.last_min_drone_animal_dist = float(min_drone_animal_dist)

        return p_drone_proximity, p_animal_proximity, p_hard_safety, hard_violation

    def _animal_state_penalty(self):
        if self.animal_count == 0:
            return 0.0, 0.0, 0.0

        avoid_count = 0
        flee_count = 0
        calm_count = 0

        for animal in self.animals:
            if animal.state == "flee":
                flee_count += 1
            elif animal.state == "avoid":
                avoid_count += 1
            else:
                calm_count += 1

        avoid_frac = avoid_count / self.animal_count
        flee_frac = flee_count / self.animal_count

        penalty = (
            1*avoid_frac + 1*flee_frac
        )

        return float(penalty), float(avoid_frac), float(flee_frac)

    def _compute_single_drone_reward(self, observations, actions):
        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0

        ALIGN_DEADZONE = 0.05
        ALIGN_EXP = 1.0
        P_VEL_EXP = 1.5
        P_THETA_EXP = 1.5
        P_DIR_CHANGE_EXP = 1.5

        p_vel = 0.0
        p_theta = 0.0
        p_dir_change = 0.0

        animal_obs_all = self._extract_animal_obs(observations)

        visible_target = np.zeros(self.drone_count, dtype=np.float32)

        for d in range(self.drone_count):
            drone_action = actions[d]
            animal_obs = animal_obs_all[d]

            in_view = animal_obs[:, 0]
            dist = animal_obs[:, 1]
            v = np.abs(animal_obs[:, 2])
            h = np.abs(animal_obs[:, 3])
            target = animal_obs[:, 7]

            is_target = target > 0.5
            target_in_view = bool(np.any(in_view[is_target] > 0.5)) if np.any(is_target) else False

            r_vis += np.sum(in_view) / max(self.animal_count, 1)

            if target_in_view:
                visible_target[d] = 1.0

                dist_term = -dist[is_target]
                r_dist += float(np.mean(dist_term))

                v_vis = np.maximum(0.0, v[is_target] - ALIGN_DEADZONE)
                h_vis = np.maximum(0.0, h[is_target] - ALIGN_DEADZONE)

                align_term = 1.0 - 0.5 * (v_vis + h_vis)
                align_term = np.clip(align_term, 0.0, 1.0)

                r_align += float(np.mean(align_term ** ALIGN_EXP))
            else:
                r_dist += -1.0

            p_vel += (
                (drone_action["vel_speed"] / (self.drones[d].max_speed + 1e-8))
                * self.config.p_vel_scale
            ) ** P_VEL_EXP

            p_theta += (
                (abs(drone_action["theta"]) / (self.drones[d].max_cam_rot + 1e-8))
                * self.config.p_theta_scale
            ) ** P_THETA_EXP

            p_dir_change += self._direction_change_penalty(d, drone_action) ** P_DIR_CHANGE_EXP

        r_vis /= max(self.drone_count, 1)
        r_dist /= max(self.drone_count, 1)
        r_align /= max(self.drone_count, 1)
        p_vel /= max(self.drone_count, 1)
        p_theta /= max(self.drone_count, 1)
        p_dir_change /= max(self.drone_count, 1)

        animal_disturbances = np.array(
            [animal.disturbance for animal in self.animals],
            dtype=np.float32
        )
        disturbance_penalty = float(np.mean(animal_disturbances)) if len(animal_disturbances) > 0 else 0.0

        target_visible_fraction = float(np.mean(visible_target)) if len(visible_target) > 0 else 0.0
        self.last_target_visible_fraction = target_visible_fraction

        disturbance_reward_tradeoff = np.clip(
            (
                self.alpha * (1.0 - disturbance_penalty)
                + (1.0 - self.alpha) * r_dist
            ) / (float(self.reward_scales[0]) + 1e-8),
            0.0,
            1.0,
        )

        r_bucket = self._bucket_balance_score()

        monitor_reward = (
            0.9 * disturbance_reward_tradeoff +
            0.1 * r_align
        )

        final_reward = (
            monitor_reward
            - p_vel
            - p_theta
            - p_dir_change
        )

        self.reward_stats["r_monitoring"] += monitor_reward
        self.reward_stats["p_disturbance"] += disturbance_penalty
        self.reward_stats["r_vis"] += r_vis
        self.reward_stats["r_dist"] += r_dist
        self.reward_stats["r_align"] += r_align
        self.reward_stats["r_bucket"] += r_bucket
        self.reward_stats["r_view_sep"] += 0.0
        self.reward_stats["r_all_view"] += 0.0
        self.reward_stats["p_not_visible"] += 0.0
        self.reward_stats["p_drone_proximity"] += 0.0
        self.reward_stats["p_animal_proximity"] += 0.0
        self.reward_stats["p_hard_safety"] += 0.0
        self.reward_stats["p_orbit_tangent"] += 0.0
        self.reward_stats["p_orbit_persistence"] += 0.0

        hard_violation = False

        if self.enable_step_logging:
            behavior_counts = {"calm": 0, "avoid": 0, "flee": 0}
            for animal in self.animals:
                behavior_counts[animal.state] += 1

            n = max(self.animal_count, 1)
            self.last_step_stats = {
                "reward": float(final_reward),
                "monitor_reward": float(monitor_reward),
                "disturbance_penalty": float(disturbance_penalty),
                "max_single_disturbance": float(self.last_max_single_disturbance),
                "target_visible_fraction": float(target_visible_fraction),
                "calm_frac": behavior_counts["calm"] / n,
                "avoid_frac": behavior_counts["avoid"] / n,
                "flee_frac": behavior_counts["flee"] / n,
                "mean_disturbance": float(np.mean([a.disturbance for a in self.animals])) if self.animals else 0.0,
                "r_vis": float(r_vis),
                "r_dist": float(r_dist),
                "r_align": float(r_align),
                "r_view_sep": 0.0,
                "r_all_view": 0.0,
                "p_not_visible": 0.0,
                "p_drone_proximity": 0.0,
                "p_animal_proximity": 0.0,
                "p_hard_safety": 0.0,
                "p_orbit_tangent": 0.0,
                "p_orbit_persistence": 0.0,
                "hard_safety_violation": 0.0,
                "min_drone_drone_dist": float(self.last_min_drone_drone_dist),
                "min_drone_animal_dist": float(self.last_min_drone_animal_dist),
            }

        return float(final_reward), float(monitor_reward), float(disturbance_penalty), bool(hard_violation)

    def _compute_multi_drone_reward(self, observations, actions):
        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0

        ALIGN_DEADZONE = 0.05
        ALIGN_EXP = 1.0
        P_VEL_EXP = 1.5
        P_THETA_EXP = 1.5
        P_DIR_CHANGE_EXP = 1.5

        p_vel = 0.0
        p_theta = 0.0
        p_dir_change = 0.0
        p_orbit_tangent = 0.0
        p_orbit_persistence = 0.0

        animal_obs_all = self._extract_animal_obs(observations)
        visible_target = np.zeros(self.drone_count, dtype=np.float32)

        for d in range(self.drone_count):
            drone_action = actions[d]
            animal_obs = animal_obs_all[d]

            in_view = animal_obs[:, 0]
            dist = animal_obs[:, 1]
            v = np.abs(animal_obs[:, 2])
            h = np.abs(animal_obs[:, 3])
            target = animal_obs[:, 7]

            is_target = target > 0.5
            target_in_view = bool(np.any(in_view[is_target] > 0.5)) if np.any(is_target) else False

            r_vis += np.sum(in_view) / max(self.animal_count, 1)

            if target_in_view:
                visible_target[d] = 1.0

                # Same as single-drone: distance is negative normalized distance.
                dist_term = -dist[is_target]
                r_dist += float(np.mean(dist_term))

                v_vis = np.maximum(0.0, v[is_target] - ALIGN_DEADZONE)
                h_vis = np.maximum(0.0, h[is_target] - ALIGN_DEADZONE)

                align_term = 1.0 - 0.5 * (v_vis + h_vis)
                align_term = np.clip(align_term, 0.0, 1.0)

                r_align += float(np.mean(align_term ** ALIGN_EXP))
            else:
                r_dist += -1.0

            p_vel += (
                (drone_action["vel_speed"] / (self.drones[d].max_speed + 1e-8))
                * self.config.p_vel_scale
            ) ** P_VEL_EXP

            p_theta += (
                (abs(drone_action["theta"]) / (self.drones[d].max_cam_rot + 1e-8))
                * self.config.p_theta_scale
            ) ** P_THETA_EXP

            p_dir_change += self._direction_change_penalty(d, drone_action) ** P_DIR_CHANGE_EXP

            orbit_tangent_penalty, orbit_persistence_penalty = self._orbit_penalty(d, drone_action)
            p_orbit_tangent += orbit_tangent_penalty * self.p_orbit_tangent_scale
            p_orbit_persistence += orbit_persistence_penalty * self.p_orbit_persistence_scale

        r_vis /= max(self.drone_count, 1)
        r_dist /= max(self.drone_count, 1)
        r_align /= max(self.drone_count, 1)

        p_vel /= max(self.drone_count, 1)
        p_theta /= max(self.drone_count, 1)
        p_dir_change /= max(self.drone_count, 1)
        p_orbit_tangent /= max(self.drone_count, 1)
        p_orbit_persistence /= max(self.drone_count, 1)

        animal_disturbances = np.array(
            [animal.disturbance for animal in self.animals],
            dtype=np.float32
        )
        disturbance_penalty = float(np.mean(animal_disturbances)) if len(animal_disturbances) > 0 else 0.0

        target_visible_fraction = float(np.mean(visible_target)) if len(visible_target) > 0 else 0.0
        self.last_target_visible_fraction = target_visible_fraction

        # Same formulation as single-drone.
        reward_scale = float(np.mean(self.reward_scales)) if len(self.reward_scales) > 0 else 1.0

        disturbance_reward_tradeoff = np.clip(
            (
                self.alpha * (1.0 - disturbance_penalty)
                + (1.0 - self.alpha) * r_dist
            ) / (reward_scale + 1e-8),
            0.0,
            1.0,
        )

        base_monitor_reward = (
            0.8 * disturbance_reward_tradeoff
            + 0.2 * r_align
        )

        # Multi-drone additions only.
        missing_fraction = 1.0 - target_visible_fraction
        p_not_visible = self.not_visible_penalty * (missing_fraction ** self.not_visible_penalty_power)

        r_all_view = self.all_view_bonus if target_visible_fraction >= 0.999 else 0.0
        r_bucket = self._bucket_balance_score()
        r_view_sep = self._view_separation_score(observations)

        monitor_reward = (
            base_monitor_reward
            + self.view_sep_weight * r_view_sep
            + r_all_view
        )

        p_drone_proximity, p_animal_proximity, p_hard_safety, hard_violation = self._compute_proximity_terms()
        p_state, avoid_frac, flee_frac = self._animal_state_penalty()

        final_reward = (
            monitor_reward
            - p_state
            - p_not_visible
            - p_vel
            - p_theta
            - p_dir_change
            - p_orbit_tangent
            - p_orbit_persistence
            - p_drone_proximity
            - p_animal_proximity
            - p_hard_safety
        )

        self.reward_stats["r_monitoring"] += monitor_reward
        self.reward_stats["p_disturbance"] += disturbance_penalty
        self.reward_stats["r_vis"] += r_vis
        self.reward_stats["r_dist"] += r_dist
        self.reward_stats["r_align"] += r_align
        self.reward_stats["r_bucket"] += r_bucket
        self.reward_stats["r_view_sep"] += r_view_sep
        self.reward_stats["r_all_view"] += r_all_view
        self.reward_stats["p_not_visible"] += p_not_visible
        self.reward_stats["p_drone_proximity"] += p_drone_proximity
        self.reward_stats["p_animal_proximity"] += p_animal_proximity
        self.reward_stats["p_hard_safety"] += p_hard_safety
        self.reward_stats["p_orbit_tangent"] += p_orbit_tangent
        self.reward_stats["p_orbit_persistence"] += p_orbit_persistence

        if self.enable_step_logging:
            behavior_counts = {"calm": 0, "avoid": 0, "flee": 0}
            for animal in self.animals:
                behavior_counts[animal.state] += 1

            n = max(self.animal_count, 1)
            self.last_step_stats = {
                "reward": float(final_reward),
                "monitor_reward": float(monitor_reward),
                "disturbance_penalty": float(disturbance_penalty),
                "max_single_disturbance": float(self.last_max_single_disturbance),
                "target_visible_fraction": float(target_visible_fraction),
                "calm_frac": behavior_counts["calm"] / n,
                "avoid_frac": behavior_counts["avoid"] / n,
                "flee_frac": behavior_counts["flee"] / n,
                "mean_disturbance": float(np.mean([a.disturbance for a in self.animals])) if self.animals else 0.0,
                "r_vis": float(r_vis),
                "r_dist": float(r_dist),
                "r_align": float(r_align),
                "r_view_sep": float(r_view_sep),
                "r_all_view": float(r_all_view),
                "p_not_visible": float(p_not_visible),
                "p_drone_proximity": float(p_drone_proximity),
                "p_animal_proximity": float(p_animal_proximity),
                "p_hard_safety": float(p_hard_safety),
                "p_orbit_tangent": float(p_orbit_tangent),
                "p_orbit_persistence": float(p_orbit_persistence),
                "hard_safety_violation": float(hard_violation),
                "min_drone_drone_dist": float(self.last_min_drone_drone_dist),
                "min_drone_animal_dist": float(self.last_min_drone_animal_dist),
            }

        return float(final_reward), float(monitor_reward), float(disturbance_penalty), bool(hard_violation)

    def compute_reward(self, observations, actions):
        if self.drone_count == 1:
            return self._compute_single_drone_reward(observations, actions)
        return self._compute_multi_drone_reward(observations, actions)

    def _check_termination(self, observations, hard_safety_violation=False):
        if hard_safety_violation:
            return True

        if self._env_steps >= self.config["max_episode_steps"]:
            return True

        animal_obs = self._extract_animal_obs(observations)
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

        reward, monitor_r, disturbance_p, hard_safety_violation = self.compute_reward(observations, actions)

        self.prev_vel_dirs = [
            drone.vel_dir.to_numpy().astype(np.float32).copy()
            for drone in self.drones
        ]

        closest_animal_dists = self._compute_closest_animal_distances()

        terminated = segment_complete or self._check_termination(
            observations,
            hard_safety_violation=hard_safety_violation,
        )
        truncated = False

        info = {
            "fov": observations,
            "reward": float(reward),
            "monitor_reward": float(monitor_r),
            "disturbance_penalty": float(disturbance_p),
            "max_single_disturbance": float(self.last_max_single_disturbance),
            "target_visible_fraction": float(self.last_target_visible_fraction),
            "hard_safety_violation": bool(hard_safety_violation),
            "min_drone_drone_dist": float(self.last_min_drone_drone_dist),
            "min_drone_animal_dist": float(self.last_min_drone_animal_dist),
            "closest_animal_distances": closest_animal_dists.copy(),
        }

        for i, dist in enumerate(closest_animal_dists):
            info[f"drone_{i}_closest_animal_distance"] = float(dist)

        if self.render_mode is not None:
            self.render(fov=observations, reward=reward)

        return observations, reward, terminated, truncated, info

    def step_log(self):
        rows = []
        closest_animal_dists = self._compute_closest_animal_distances()

        for animal in self.animals:
            rows.append({
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
                "max_single_disturbance": float(getattr(animal, "max_single_disturbance", 0.0)),
                "bucket_front": self.view_bucket_counts[animal.id, 0] if animal.id < self.animal_count else "",
                "bucket_left": self.view_bucket_counts[animal.id, 1] if animal.id < self.animal_count else "",
                "bucket_back": self.view_bucket_counts[animal.id, 2] if animal.id < self.animal_count else "",
                "bucket_right": self.view_bucket_counts[animal.id, 3] if animal.id < self.animal_count else "",
                "is_standoff": "",
                "standoff_target_distance": "",
                "closest_animal_distance": "",
                "target_visible_fraction": "",
                "min_drone_drone_dist": "",
                "min_drone_animal_dist": "",
                "hard_safety_violation": "",
            })

        for i, drone in enumerate(self.drones):
            rows.append({
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
                "max_single_disturbance": float(self.last_max_single_disturbance),
                "bucket_front": "",
                "bucket_left": "",
                "bucket_back": "",
                "bucket_right": "",
                "is_standoff": "",
                "standoff_target_distance": "",
                "closest_animal_distance": float(closest_animal_dists[i]),
                "target_visible_fraction": float(self.last_target_visible_fraction),
                "min_drone_drone_dist": float(self.last_min_drone_drone_dist),
                "min_drone_animal_dist": float(self.last_min_drone_animal_dist),
                "hard_safety_violation": float(self.last_hard_safety_violation),
                **self.last_step_stats,
            })

        return rows