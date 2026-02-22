import random
import numpy as np
from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .disturbance import disturbance_gain
from .immutables import Behavior, MovementDim

class Env:
    def __init__(self, config, render_mode=None, seed=42):
        self.set_seed(seed)
        self.render_mode = render_mode
        self.config = config
        self.viewer = Viewer(config)
        
        self.animal_count = config["animal"]["env"]["count"]
        self.animals = [Animal(config=config, 
                               behaviors=Behavior, 
                               movement_dims=MovementDim,
                               )
                        for _ in range(self.animal_count)]
        
        self.drones = []
        for drone_type in ["small", "large"]:
            count = config["drone"][drone_type]["count"]
            for _ in range(count):
                self.drones.append(Drone(config=config, 
                                         d_type=drone_type))
            
        self.drone_count = len(self.drones)
        self._env_steps = 0

    def _init_animal(self):
        # randomization using animal.rng, animal seed decides spawn location, spawn heading and behaviour
        for i, animal in enumerate(self.animals):
            animal.disturbance = 0.0
            animal.escape_dir = np.zeros(3, dtype=np.float32)
            animal.vel_dir = Vector().random_unit(dim=self.config["animal"]["init"]["movement_dim"], rng=self.env_rng)
            animal.vel_speed = random.uniform(animal.min_speed, animal.max_speed)
            spawn_dir = Vector().random_unit(dim=self.config["animal"]["init"]["movement_dim"], rng=self.env_rng)
            radius = self.env_rng.uniform(0, self.config["animal"]["init"]["max_spawn_radius"])
            animal.pos = spawn_dir.scale(radius)

    def _init_drone(self):
            for i in range(self.drone_count):
                # Use modulo to wrap around the animal list if drones > animals
                # If drone_count is 5 and animal_count is 2: 
                # Drones 0, 2, 4 get Animal 0 | Drones 1, 3 get Animal 1
                animal = self.animals[i % self.animal_count]
                drone = self.drones[i]

                animal_pos = animal.pos.getter()    

                # Reset and randomize view direction
                drone.view_dir.unit()
                drone.view_dir.rotate_z(self.env_rng.uniform(0, 360))

                view_dir = drone.view_dir.getter()

                # Calculate position: move backwards from animal by spawn_dist
                inv_scale_view_dir = view_dir.scale(-drone.spawn_dist)
                new_drone_pos = animal_pos.add(inv_scale_view_dir)

                drone.pos.setter(new_drone_pos)

                # Randomize initial speed
                drone.vel_speed = self.env_rng.uniform(
                    drone.min_speed, 
                    drone.max_speed
                )

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
        if seed == None: # for init
            self.curr_episode_seed = self.next_episode_seed
        else:
            self.curr_episode_seed = seed
        
        self.seeds = np.random.SeedSequence(self.curr_episode_seed).spawn(2)
        self.env_rng = np.random.default_rng(self.seeds[0])
        self.sample_action_rng = np.random.default_rng(self.seeds[1]) # sample action needs its own rng, otherwise it can influence env

        self.next_episode_seed = self.env_rng.integers(0, np.iinfo(np.int32).max) # generate next seed immediately -> episode length has no influence on next seed
        self.encounter_map_seed = self.env_rng.integers(0, np.iinfo(np.int32).max) # generate next seed immediately -> episode length has no influence on next seed

    def reset(self, seed=None):
        """
        Reinitializes drone and animal initial positions according to config.
        Returns observations, the animals and drones.
        
        :param seed: makes every episode the same
        """

        self._env_steps = 0
        self.set_seed(seed)

        self._init_animal()
        self._init_drone()

        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        info = {}
        return observations, info
    
    def package_actions(self, actions):
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
            drone.theta = action["theta"]
            drone.enforce_cam_rot()
            drone.view_dir.rotate_z(drone.theta)
            drone.reset_theta()

    def _step_animal(self):

        for animal in self.animals:

            D = animal.disturbance

            if D > 0.7:
                # FULL ESCAPE
                animal.vel_dir.setter(Vector(*animal.escape_dir))
                animal.vel_speed = animal.max_speed

            elif D > 0.4:
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
            animal.update_pos()
            animal.enforce_position()

    def in_FoV(self, drones, animals):
        """
        x: view dir (forward)
        y: right
        z: up
        """

        in_FoV_obs = []
        world_z = np.array([0, 0, 1], dtype=np.float32)

        for drone in drones:

            drone_obs = []

            # View direction (forward axis)
            x = drone.view_dir.to_numpy()
            x = x / (np.linalg.norm(x) + 1e-8)

            view_x, view_y, view_z = x

            # Altitude sensor
            altitude = drone.pos.to_numpy()[2]
            altitude_norm = altitude / (drone.max_altitude + 1e-8)

            # Camera basis
            y = np.cross(world_z, x)
            y = y / (np.linalg.norm(y) + 1e-8)

            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-8)

            v_max = np.deg2rad(drone.ver_angle / 2)
            h_max = np.deg2rad(drone.hor_angle / 2)

            for animal in animals:

                drone_pos = drone.pos.to_numpy()
                animal_pos = animal.pos.to_numpy()

                rel_vec = animal_pos - drone_pos
                distance = np.linalg.norm(rel_vec)

                drone_to_animal_vec = (
                    rel_vec / distance if distance >= 1e-8
                    else np.zeros(3)
                )

                cx = np.dot(drone_to_animal_vec, x)
                cy = np.dot(drone_to_animal_vec, y)
                cz = np.dot(drone_to_animal_vec, z)

                v_angle = np.arctan2(cz, cx)
                h_angle = np.arctan2(cy, cx)

                in_view = (
                    cx > 0 and
                    abs(v_angle) <= v_max and
                    abs(h_angle) <= h_max and
                    distance <= drone.view_range
                )

                if in_view:

                    dist_norm = distance / drone.view_range
                    v_norm = v_angle / v_max
                    h_norm = h_angle / h_max

                    drone_obs.append([
                        1.0,
                        dist_norm,
                        v_norm,
                        h_norm,
                        view_x,
                        view_y,
                        view_z,
                        altitude_norm
                    ])

                else:

                    drone_obs.append([
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        view_x,
                        view_y,
                        view_z,
                        altitude_norm
                    ])

            in_FoV_obs.append(np.array(drone_obs, dtype=np.float32))

        return np.array(in_FoV_obs, dtype=np.float32)

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
    
    def _compute_geometry(self):

        geometry = []

        drone_positions = [d.pos.to_numpy() for d in self.drones]
        animal_positions = [a.pos.to_numpy() for a in self.animals]

        for drone_pos in drone_positions:

            drone_geom = []

            for animal_pos in animal_positions:

                rel_vec = animal_pos - drone_pos
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
        """
        Builds FoV observations using precomputed geometry.
        """

        in_FoV_obs = []
        world_z = np.array([0, 0, 1], dtype=np.float32)

        for d, drone in enumerate(self.drones):

            drone_obs = []

            # forward axis
            x = drone.view_dir.to_numpy()
            x = x / (np.linalg.norm(x) + 1e-8)

            view_x, view_y, view_z = x

            altitude = drone.pos.to_numpy()[2]
            altitude_norm = altitude / (drone.max_altitude + 1e-8)

            # camera basis
            y = np.cross(world_z, x)
            y = y / (np.linalg.norm(y) + 1e-8)

            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-8)

            v_max = np.deg2rad(drone.ver_angle / 2)
            h_max = np.deg2rad(drone.hor_angle / 2)

            for a in range(self.animal_count):

                rel_unit = geometry[d][a]["dir_unit"]
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

                if in_view:
                    drone_obs.append([
                        1.0,
                        distance / drone.view_range,
                        v_angle / v_max,
                        h_angle / h_max,
                        view_x, view_y, view_z,
                        altitude_norm
                    ])
                else:
                    drone_obs.append([
                        0.0, 1.0, 0.0, 0.0,
                        view_x, view_y, view_z,
                        altitude_norm
                    ])

            in_FoV_obs.append(np.array(drone_obs, dtype=np.float32))

        return np.array(in_FoV_obs, dtype=np.float32)
    
    def _compute_disturbance(self, geometry):


        for a, animal in enumerate(self.animals):

            instant_disturbance = 0.0
            escape_vec = np.zeros(3, dtype=np.float32)

            for d, drone in enumerate(self.drones):

                rel_vec = geometry[d][a]["rel_vec"]
                distance = geometry[d][a]["distance"]

                gain = disturbance_gain(rel_vec) * drone.disturbance_mult
                instant_disturbance += gain

                if distance > 1e-8:
                    escape_vec += (-rel_vec / distance) * gain

            instant_disturbance = np.clip(instant_disturbance, 0.0, 1.0)
            if instant_disturbance > animal.disturbance:
                beta = 0.35   # fast panic
            else:
                beta = 0.05   # slow recovery

            # INERTIA
            animal.disturbance = (
                (1.0 - beta) * animal.disturbance +
                beta * instant_disturbance
            )

            animal.disturbance = np.clip(animal.disturbance, 0.0, 1.0)

            # escape direction
            norm = np.linalg.norm(escape_vec)
            if norm > 1e-8:
                animal.escape_dir = escape_vec / norm
            else:
                animal.escape_dir = animal.vel_dir.to_numpy()
        
    def compute_reward(self, observations):

        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0

        # MONITORING QUALITY
        for d in range(self.drone_count):

            drone_obs = observations[d]

            in_view = drone_obs[:, 0]
            dist = drone_obs[:, 1]
            v = np.abs(drone_obs[:, 2])
            h = np.abs(drone_obs[:, 3])

            visible = in_view == 1.0

            if np.any(visible):
                r_vis   += np.mean(in_view)
                r_dist  += np.mean(1.0 - dist[visible])
                r_align += np.mean(1.0 - 0.5 * (v[visible] + h[visible]))

        r_vis   /= self.drone_count
        r_dist  /= self.drone_count
        r_align /= self.drone_count

        # DISTURBANCE (STATE-BASED)
        animal_disturbances = np.array(
            [animal.disturbance for animal in self.animals],
            dtype=np.float32
        )

        # mean stress in herd
        disturbance_penalty = np.clip(
            np.mean(animal_disturbances),
            0.0,
            1.0
        )

        # FINAL REWARD
        monitor_reward = (
            0.6 * r_align +
            0.3 * r_dist +
            0.1 * r_vis
        )

        alpha = 0.3

        final_reward = (
            (1 - alpha) * monitor_reward +
            alpha * (1.0 - disturbance_penalty)
        )

        return final_reward

    def step(self, actions):
        self._env_steps += 1

        terminated = False
        truncated = False

        if self._env_steps >= self.config["max_episode_steps"]:
            terminated = True

        actions = self.package_actions(actions)

        # 1. apply actions
        self._step_drone(self.drones, actions)

        # 2. animals react to new drone positions
        geometry = self._compute_geometry()
        self._compute_disturbance(geometry)
        self._step_animal()

        # 3. observe resulting state
        geometry = self._compute_geometry()
        observations = self._build_observations(geometry)

        # 4. compute reward FROM RESULT
        reward = self.compute_reward(observations)

        info = {
            "fov": observations
        }

        if self.render_mode is not None:
            self.render(fov=observations, reward=reward)
            
        return observations, reward, terminated, truncated, info