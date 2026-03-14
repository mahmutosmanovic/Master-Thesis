import numpy as np
from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .disturbance import disturbance_gain, disturbance_gain_alt
from .resource_map import ResourceMap
from .immutables import MovementDim
from .behaviors import CRW_CFG


class Env:
    def __init__(self, config, render_mode=None, seed=42):
        self.set_seed(seed)
        self.render_mode = render_mode
        self.config = config
        self.viewer = Viewer(config)
        self.resource_map = None

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
        }

        self.total_state_steps = 0
        self.disturbance_sum = 0.0
        self.drone_feature_count = config.model.space.drone_features
        self.animal_feature_count = config.model.space.animal_features

        # needs proper fix
        self.scale_factor = sum(
            [drone_type.count * drone_type.disturbance_mult for drone_type in self.config.drone.values()]
        )

        self._env_steps = 0
        self.episode = -1

    def _create_resource_map(self):
        if type(self.config["animal"]["init"]["behavior"]) == CRW_CFG:
            return None
        else:
            return ResourceMap(config=self.config, seed=self.resource_map_seed)

    def _init_animal(self):
        # randomization using animal.rng, animal seed decides spawn location, spawn heading and behaviour
        for i, animal in enumerate(self.animals):
            animal.disturbance = 0.0
            animal.arousal = 0.0
            animal.field_boost = 0.0
            animal.state = "calm"
            animal.escape_dir = np.zeros(3, dtype=np.float32)
            animal.resource_map = self.resource_map
            if getattr(animal.behavior, "handles_spawn", False):
                animal.behavior.reset(animal=animal, rng=self.env_rng)
            else:
                animal.vel_dir = Vector().random_unit(dim=self.movement_dim, rng=self.env_rng)
                animal.vel_speed = self.env_rng.uniform(animal.min_speed, animal.max_speed)
                spawn_dir = Vector().random_unit(dim=self.movement_dim, rng=self.env_rng)
                radius = self.env_rng.uniform(0, self.config["animal"]["init"]["max_spawn_radius"])
                animal.pos = spawn_dir.scale(radius)
                animal.behavior.reset()

    def _update_arousal(self, arousal, disturbance):
        alpha = self.config.animal.env.arousal_alpha
        arousal = (1.0 - alpha) * arousal + alpha * disturbance
        return np.clip(arousal, 0.0, 1.0)

    def _state_from_arousal(self, arousal):
        if arousal > self.config.animal.env.flee_threshold:
            return "flee"
        elif arousal > self.config.animal.env.avoid_threshold:
            return "avoid"
        return "calm"


    def _update_field_boost(self, boost, state):
        if state == "flee":
            target = 10.0
        elif state == "avoid":
            target = 3.0
        else:
            target = 0.0

        rise = self.config.animal.env.boost_rise
        decay = self.config.animal.env.boost_decay
        alpha = rise if target > boost else decay

        boost = boost + alpha * (target - boost)
        
        noise_std = self.config.animal.env.boost_noise_std
        boost += self.env_rng.normal(0.0, noise_std)

        return np.clip(boost, 0.0, 10.0)

    def _init_drone(self):
        for i in range(self.drone_count):
            # Use modulo to wrap around the animal list if drones > animals
            animal = self.animals[i % self.animal_count]
            drone = self.drones[i]

            animal_pos = animal.pos.getter()

            # Reset and randomize view direction
            drone.view_dir.unit()
            drone.view_dir.rotate_z(self.env_rng.uniform(0, 360))

            view_dir = drone.view_dir.getter()

            # Calculate position: move backwards from animal by spawn_dist
            inv_scale_view_dir = view_dir.scale(-self.env_rng.uniform(*drone.spawn_dist))
            new_drone_pos = animal_pos.add(inv_scale_view_dir)

            drone.pos.setter(new_drone_pos)

            # Randomize initial speed
            drone.vel_speed = self.env_rng.uniform(drone.min_speed, drone.max_speed)

    def sample_action(self):
        """
        Returns a flat action array compatible with env.step().

        For each drone:
            [vx, vy, vz, vel_speed, theta]
        """

        actions = []

        for drone in self.drones:
            # random 3D unit direction
            vel_dir = Vector().random_unit(dim=MovementDim.THREE_D, rng=self.sample_action_rng)
            vx, vy, vz = vel_dir.to_numpy()

            # random speed within limits
            vel_speed = self.sample_action_rng.uniform(drone.min_speed, drone.max_speed)

            # random camera rotation
            theta = self.sample_action_rng.triangular(
                -drone.max_cam_rot,
                0,
                drone.max_cam_rot
            )

            actions.extend([vx, vy, vz, vel_speed, theta])

        return np.array(actions, dtype=np.float32)

    def set_seed(self, seed):
        if seed is None:  # for init
            self.curr_episode_seed = self.next_episode_seed
        else:
            self.curr_episode_seed = seed

        self.seeds = np.random.SeedSequence(self.curr_episode_seed).spawn(2)
        self.env_rng = np.random.default_rng(self.seeds[0])
        self.sample_action_rng = np.random.default_rng(self.seeds[1])  # separate RNG

        self.next_episode_seed = self.env_rng.integers(0, np.iinfo(np.int32).max)
        self.resource_map_seed = int(self.env_rng.integers(0, np.iinfo(np.int32).max))

    def reset(self, seed=None):
        """
        Reinitializes drone and animal initial positions according to config.
        Returns observations, info.

        :param seed: makes every episode the same
        """

        self._env_steps = 0
        self.episode += 1
        self.state_counts = {"calm": 0, "avoid": 0, "flee": 0}
        self.reward_stats = {
            "r_monitoring": 0,
            "p_disturbance": 0,
            "r_vis": 0,
            "r_dist": 0,
            "r_align": 0,
        }
        self.total_state_steps = 0
        self.disturbance_sum = 0.0

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

            # camera basis
            x, y, z = self._camera_basis(drone)

            # convert camera-frame velocity to world-frame
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
        for drone, action in zip(drones, actions):
            # movement
            drone.vel_dir.setter(action["vel_dir"])
            drone.vel_dir.unit()

            drone.vel_speed = action["vel_speed"]
            drone.enforce_speed()

            drone.update_pos()
            drone.enforce_position()

            # camera
            alpha = 0.8
            drone.theta = alpha * drone.theta + (1.0 - alpha) * action["theta"]
            drone.enforce_cam_rot()
            drone.view_dir.rotate_z(drone.theta)

    def _step_animal(self):
        segment_complete = False
        for animal in self.animals:
            if animal.behavior.can_flee:
                state = animal.state
                if state == "flee":
                    # FULL ESCAPE
                    animal.vel_dir.setter(Vector(*animal.escape_dir))
                    animal.vel_speed = animal.max_speed

                elif state == "avoid":
                    # AVOIDANCE BLEND
                    animal.update_vel(rng=self.env_rng)

                    base = animal.vel_dir.to_numpy()
                    flee = animal.escape_dir

                    blended = 0.5 * base + 0.5 * flee
                    animal.vel_dir.setter(Vector(*blended))

                else:
                    # NORMAL BEHAVIOUR
                    animal.update_vel(rng=self.env_rng)

                animal.enforce_speed()
                D = animal.disturbance
            else:
                # track following behaviour, no speed enforcement!
                D = animal.disturbance
                state = "calm"
                a_segment_complete = animal.update_vel(rng=self.env_rng)
                if segment_complete is False and a_segment_complete:
                    segment_complete = True

            animal.state = state
            self.state_counts[state] += 1
            self.total_state_steps += 1
            self.disturbance_sum += D

            animal.update_pos()
            animal.enforce_position()

        return segment_complete

    def get_behavior_stats(self):
        if self.total_state_steps == 0:
            return None

        return {
            "calm_frac": self.state_counts["calm"] / self.total_state_steps,
            "avoid_frac": self.state_counts["avoid"] / self.total_state_steps,
            "flee_frac": self.state_counts["flee"] / self.total_state_steps,
        }

    def get_reward_stats(self):
        if self.total_state_steps == 0:
            return None

        return {
            "r_monitoring": self.reward_stats["r_monitoring"] / self.total_state_steps,
            "p_disturbance": self.reward_stats["p_disturbance"] / self.total_state_steps,
            "r_vis": self.reward_stats["r_vis"] / self.total_state_steps,
            "r_dist": self.reward_stats["r_dist"] / self.total_state_steps,
            "r_align": self.reward_stats["r_align"] / self.total_state_steps,
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

                animal_vel = animal.vel_dir.to_numpy().astype(np.float32) * float(animal.vel_speed)

                if in_view:
                    animal_features.extend([
                        1.0,
                        distance / drone.view_range,
                        v_angle / (v_max + 1e-8),
                        h_angle / (h_max + 1e-8),
                        animal_vel[0],
                        animal_vel[1],
                        animal_vel[2],
                    ])
                else:
                    animal_features.extend([
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        animal_vel[0],
                        animal_vel[1],
                        animal_vel[2],
                    ])

            obs_all.append(np.array(drone_features + animal_features, dtype=np.float32))

        return np.array(obs_all, dtype=np.float32)

    def _compute_disturbance(self, geometry):
        for a, animal in enumerate(self.animals):
            escape_vec = np.zeros(3, dtype=np.float32)
            animal.disturbance = 0.0
            disturbances = []
            escape_vecs = []

            # gather disturbances
            for d, drone in enumerate(self.drones):
                rel_vec = geometry[d][a]["rel_vec"]
                distance = geometry[d][a]["distance"]

                if distance > 1e-8:
                    unit_escape_vec = -rel_vec / distance
                else:
                    unit_escape_vec = np.zeros(3, dtype=np.float32)

                escape_vecs.append(unit_escape_vec)

                gain = disturbance_gain_alt(rel_vec, field_boost=animal.field_boost) * drone.disturbance_mult

                disturbances.append(gain)

            sorted_idx = np.argsort(disturbances)[::-1]
            for i in range(len(disturbances)):
                if animal.disturbance >= 1:
                    break

                idx = sorted_idx[i]
                disturbance = disturbances[idx]
                influence = (1 - animal.disturbance) * disturbance
                animal.disturbance += influence
                escape_vec += escape_vecs[idx] * influence

            # escape direction
            norm = np.linalg.norm(escape_vec)
            if norm > 1e-8:
                animal.escape_dir = escape_vec / norm
            else:
                animal.escape_dir = animal.vel_dir.to_numpy()

            animal.arousal = self._update_arousal(animal.arousal, animal.disturbance)
            animal.state = self._state_from_arousal(animal.arousal)
            animal.field_boost = self._update_field_boost(animal.field_boost, animal.state)

    def compute_reward(self, observations, actions):
        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0

        visible_any = False

        ALIGN_DEADZONE = 0.10
        DIST_EXP = 2.0
        ALIGN_EXP = 2.0

        for d in range(self.drone_count):
            drone_obs = observations[d]

            # [in_view, dist_norm, v_angle, h_angle, velx, vely, velz]
            animal_obs = drone_obs[self.drone_feature_count:].reshape(self.animal_count, self.animal_feature_count)

            in_view = animal_obs[:, 0]
            dist = animal_obs[:, 1]
            v = np.abs(animal_obs[:, 2])
            h = np.abs(animal_obs[:, 3])

            visible = in_view == 1.0

            # visibility reward
            r_vis += np.sum(in_view) / self.animal_count

            if np.any(visible):
                visible_any = True

                # distance shaping
                dist_term = 1.0 - dist[visible]
                r_dist += np.mean(dist_term ** DIST_EXP)

                # alignment with dead-zone
                v_vis = np.maximum(0.0, v[visible] - ALIGN_DEADZONE)
                h_vis = np.maximum(0.0, h[visible] - ALIGN_DEADZONE)

                align_term = 1.0 - 0.5 * (v_vis + h_vis)
                align_term = np.clip(align_term, 0.0, 1.0)

                r_align += np.mean(align_term ** ALIGN_EXP)

        # normalize across drones
        r_vis /= self.drone_count
        r_dist /= self.drone_count
        r_align /= self.drone_count

        # disturbance/state penalty
        p_disturbance = float(np.mean([animal.disturbance for animal in self.animals]))

        rew_components = [r_dist, r_align]
        monitor_reward = sum(rew_components) / len(rew_components)

        final_reward = monitor_reward - 1.5*p_disturbance

        # penalty if nothing visible
        if not visible_any:
            final_reward -= 0.2

        self.reward_stats["r_monitoring"] += monitor_reward
        self.reward_stats["p_disturbance"] += p_disturbance
        self.reward_stats["r_dist"] += r_dist
        self.reward_stats["r_align"] += r_align
        self.reward_stats["r_vis"] += r_vis

        return final_reward

    def _check_termination(self, observations):
        if self._env_steps >= self.config["max_episode_steps"]:
            return True

        animal_obs = observations[:, 4:].reshape(
            self.drone_count,
            self.animal_count,
            self.animal_feature_count
        )

        visible = animal_obs[:, :, 0] == 1.0
        animal_visible = np.any(visible, axis=0)

        visible_count = np.sum(animal_visible)

        # terminate episode if 50% of animals are not visible
        if visible_count < (self.animal_count * 0.5):
            return True

        return False

    def step(self, actions):
        self._env_steps += 1

        actions = self.package_actions(actions)

        # 1. apply actions
        self._step_drone(self.drones, actions)

        # 2. animals react to new drone positions
        geometry = self._compute_geometry()
        self._compute_disturbance(geometry)
        segment_complete = self._step_animal()

        # 3. observe resulting state
        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        # 4. compute reward FROM RESULT
        reward = self.compute_reward(observations, actions)

        # 5. termination and truncation
        terminated = segment_complete or self._check_termination(observations)
        truncated = False

        info = {
            "fov": observations,
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
                "arousal": animal.arousal,
                "field_boost": animal.field_boost,
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
                "arousal": "",
                "field_boost": "",
            })

        return rows