import numpy as np
from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .disturbance import disturbance_gain, animal_axis_gain
from .resource_map import ResourceMap
from .immutables import MovementDim
from .behaviors import CRW_CFG


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
            "r_monitoring": 0,
            "p_disturbance": 0,
            "r_vis": 0,
            "r_dist": 0,
            "r_align": 0,
            "r_bucket": 0,
        }

        self.disturbance_sum = 0.0

        self.scale_factor = sum(
            [drone_type.count * drone_type.disturbance_mult for drone_type in self.config.drone.values()]
        )

        self._env_steps = 0
        self.episode = -1

        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.prev_vel_dirs = None

        self.enable_step_logging = False
        self.last_step_stats = {}

        self.out_of_view_steps = 0

    def _create_resource_map(self):
        if type(self.config["animal"]["init"]["behavior"]) == CRW_CFG:
            return None
        else:
            return ResourceMap(config=self.config, seed=self.resource_map_seed)

    def _init_animal(self):
        for i, animal in enumerate(self.animals):
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
                drone.max_cam_rot
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
            "r_monitoring": 0,
            "p_disturbance": 0,
            "r_vis": 0,
            "r_dist": 0,
            "r_align": 0,
            "r_bucket": 0,
            "episode_progress": 0
        }

        if self.enable_step_logging:
            self.last_step_stats = {
                "reward": 0.0,
                "calm_frac": 0.0,
                "avoid_frac": 0.0,
                "flee_frac": 0.0,
                "mean_disturbance": 0.0,
                "r_monitoring": 0.0,
                "p_disturbance": 0.0,
                "r_vis": 0.0,
                "r_dist": 0.0,
                "r_align": 0.0,
            }

        self.disturbance_sum = 0.0

        self.view_bucket_counts = np.zeros((self.animal_count, 4), dtype=np.float32)
        self.view_bucket_totals = np.zeros(self.animal_count, dtype=np.float32)

        self.set_seed(seed)
        self.resource_map = self._create_resource_map()

        self._init_animal()
        self._init_drone()

        self.prev_vel_dirs = [
            drone.vel_dir.to_numpy().astype(np.float32).copy()
            for drone in self.drones
        ]

        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        info = {}
        return observations, info

    def reset_episode_id(self):
        self.episode = -1

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

            package_action = {
                "vel_dir": Vector(drone_actions[0], drone_actions[1], drone_actions[2]),
                "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                "theta": norm_theta * max_cam_rot
            }

            packaged_actions.append(package_action)

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

            world_vec = (
                v_forward * x +
                v_right * y +
                v_up * z
            )

            vel_dir = Vector(*world_vec)

            package_action = {
                "vel_dir": vel_dir,
                "vel_speed": ((norm_speed + 1.0) * 0.5 * (max_speed - min_speed) + min_speed),
                "theta": norm_theta * max_cam_rot
            }

            packaged_actions.append(package_action)

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

            prev_view = drone.view_dir.to_numpy().astype(np.float32).copy()
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
                if segment_complete is False and a_segment_complete:
                    segment_complete = True

            animal.state = state
            self.state_counts[state] += 1
            self.disturbance_sum += D

            animal.update_pos()
            animal.enforce_position()

        return segment_complete

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
            "r_vis": self.reward_stats["r_vis"] / self._env_steps,
            "r_dist": self.reward_stats["r_dist"] / self._env_steps,
            "r_align": self.reward_stats["r_align"] / self._env_steps,
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
            reward=reward
        )

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

    def _animal_bucket_fractions(self, animal_idx):
        total = self.view_bucket_totals[animal_idx]
        if total <= 1e-8:
            return np.zeros(4, dtype=np.float32)
        return self.view_bucket_counts[animal_idx] / total

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
        drone_feat_dim = self.config.model.space.drone_features
        animal_feat_dim = self.config.model.space.animal_features
        animal_obs = observations[:, drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim].reshape(self.drone_count, self.animal_count, animal_feat_dim)

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
        world_z = np.array([0, 0, 1], dtype=np.float32)

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
                    cx > 0 and
                    abs(v_angle) <= v_max and
                    abs(h_angle) <= h_max and
                    distance <= drone.view_range
                )

                animal_vel = (
                    animal.vel_dir.to_numpy() *
                    (animal.vel_speed / (animal.max_speed + 1e-8))
                )

                v_cam_x = np.dot(animal_vel, x)
                v_cam_y = np.dot(animal_vel, y)
                v_cam_z = np.dot(animal_vel, z)

                if in_view:
                    animal_features.extend([
                        1.0,
                        distance / drone.view_range,
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
                        0.0,
                    ])

            obs_all.append(
                np.array(
                    drone_features + animal_features,
                    dtype=np.float32
                )
            )

        return np.array(obs_all, dtype=np.float32)

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
                    self.config
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

    def _direction_change_penalty(self, drone_idx, drone_action):
        if self.prev_vel_dirs is None:
            return 0.0

        curr_dir = drone_action["vel_dir"].to_numpy().astype(np.float32)
        curr_norm = np.linalg.norm(curr_dir)

        if curr_norm > 1e-8:
            curr_dir = curr_dir / curr_norm
        else:
            return 0.0

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

    def compute_reward(self, observations, actions):
        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0
        r_cover = 0.0

        visible_target = np.array([0.0]*self.drone_count)

        ALIGN_DEADZONE = 0.05
        DIST_EXP = 1.0
        ALIGN_EXP = 1.0

        p_vel = 0.0
        p_theta = 0.0
        p_dir_change = 0.0

        for d in range(self.drone_count):
            drone_obs = observations[d]
            drone_action = actions[d]

            # 11 features per animal:
            # [in_view, dist_norm, v_angle, h_angle, velx, vely, velz]
            drone_feat_dim = self.config.model.space.drone_features
            animal_feat_dim = self.config.model.space.animal_features
            animal_obs = drone_obs[drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim].reshape(self.animal_count, animal_feat_dim)

            in_view = animal_obs[:, 0]
            dist = animal_obs[:, 1]
            v = np.abs(animal_obs[:, 2])
            h = np.abs(animal_obs[:, 3])
            target = animal_obs[:, 7]

            # visible = in_view == 1.0
            is_target = target > 0.5

            r_vis += np.sum(in_view) / self.animal_count

            # visible_other = np.any(np.logical_xor(visible, is_target))

            if np.any(is_target):
                visible_target[d] = 1.0

                target_dist = 0.45
                dist_err = np.abs(dist[is_target] - target_dist) / target_dist
                dist_term_vis = np.clip(1.0 - dist_err, 0.0, 1.0)
                r_dist += np.mean(dist_term_vis ** DIST_EXP)

                v_vis = np.maximum(0.0, v[is_target] - ALIGN_DEADZONE)
                h_vis = np.maximum(0.0, h[is_target] - ALIGN_DEADZONE)

                align_term_vis = 1.0 - 0.5 * (v_vis + h_vis)
                align_term_vis = np.clip(align_term_vis, 0.0, 1.0)
                r_align += np.mean(align_term_vis ** ALIGN_EXP)

                cover_terms = []
                visible_indices = np.where(is_target)[0]

                drone_pos = np.asarray(self.drones[d].pos.to_numpy(), dtype=np.float32)

                for local_idx, a_idx in enumerate(visible_indices):
                    animal_pos = np.asarray(self.animals[a_idx].pos.to_numpy(), dtype=np.float32)
                    animal_vel_dir = np.asarray(self.animals[a_idx].vel_dir.to_numpy(), dtype=np.float32)

                    world_dist_vec = drone_pos - animal_pos

                    # side-on high, front/back low
                    cover_raw = 1.0 - animal_axis_gain(world_dist_vec, animal_vel_dir)

                    dist_q = float(dist_term_vis[local_idx] ** DIST_EXP)
                    align_q = float(align_term_vis[local_idx] ** ALIGN_EXP)

                    # only reward coverage when observation quality is already decent
                    cover_term = cover_raw * np.sqrt(dist_q * align_q)
                    cover_terms.append(cover_term)

                if len(cover_terms) > 0:
                    r_cover += float(np.mean(cover_terms))

            p_vel += (
                (drone_action["vel_speed"] / (self.drones[d].max_speed + 1e-8))
                * self.config.p_vel_scale
            )

            p_theta += (
                (abs(drone_action["theta"]) / (self.drones[d].max_cam_rot + 1e-8))
                * self.config.p_theta_scale
            )

            p_dir_change += self._direction_change_penalty(d, drone_action)

        r_vis /= self.drone_count
        r_dist /= self.drone_count
        r_align /= self.drone_count
        r_cover /= self.drone_count
        p_vel /= self.drone_count
        p_theta /= self.drone_count
        p_dir_change /= self.drone_count

        animal_disturbances = np.array(
            [animal.disturbance for animal in self.animals],
            dtype=np.float32
        )
        disturbance_penalty = float(np.mean(animal_disturbances)) * self.disturb_scale

        r_bucket = self._bucket_balance_score()

        frac_visible = visible_target.sum() / self.drone_count
        # bonus scaling by number of drones that lost their target
        visible_bonus = frac_visible * 0.05

        monitor_reward = (
            0.85 * r_dist +
            0.85 * r_align +
            0.65 * r_cover +
            visible_bonus
        )

        final_reward = (
            monitor_reward
            - disturbance_penalty
            - p_vel
            - p_theta
            - p_dir_change
        )

        # penalty scaling by number of drones that lost their target
        final_reward -= (1.0 - frac_visible) * float(self.config.not_visible_penalty)

        self.reward_stats["r_monitoring"] += monitor_reward
        self.reward_stats["p_disturbance"] += disturbance_penalty
        self.reward_stats["r_vis"] += r_vis
        self.reward_stats["r_dist"] += r_dist
        self.reward_stats["r_align"] += r_align
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
                "r_vis": float(r_vis),
                "r_dist": float(r_dist),
                "r_align": float(r_align),
                "r_cover": float(r_cover),
            }

        return float(final_reward), monitor_reward, disturbance_penalty

    def _check_termination(self, observations):
        if self._env_steps >= self.config["max_episode_steps"]:
            return True

        drone_feat_dim = self.config.model.space.drone_features
        animal_feat_dim = self.config.model.space.animal_features
        animal_obs = observations[:, drone_feat_dim:drone_feat_dim + self.animal_count * animal_feat_dim].reshape(self.drone_count, self.animal_count, animal_feat_dim)

        visible = animal_obs[:, :, 7] > 0.5
        target_visible = np.any(visible, axis=0)

        target_visible_count = np.sum(target_visible)

        # terminate episode if 50% of drones have lost their target animal and grace period runs out
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
        observations = self._build_observations(geometry)

        reward, monitor_r, disturbance_p = self.compute_reward(observations, actions)

        self.prev_vel_dirs = [
            drone.vel_dir.to_numpy().astype(np.float32).copy()
            for drone in self.drones
        ]

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

    def step_log(self):
        rows = []

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
                "bucket_front": self.view_bucket_counts[animal.id, 0] if animal.id < self.animal_count else "",
                "bucket_left": self.view_bucket_counts[animal.id, 1] if animal.id < self.animal_count else "",
                "bucket_back": self.view_bucket_counts[animal.id, 2] if animal.id < self.animal_count else "",
                "bucket_right": self.view_bucket_counts[animal.id, 3] if animal.id < self.animal_count else "",
            })

        for drone in self.drones:
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
                "bucket_front": "",
                "bucket_left": "",
                "bucket_back": "",
                "bucket_right": "",
                **self.last_step_stats
            })

        return rows