import random
import numpy as np
from .vec import Vector
from .viewer import Viewer
from .entity import Drone, Animal
from .immutables import Behavior, MovementDim

class Env:
    def __init__(self, config, render_mode=None, seed=42):
        self.set_seed(seed)

        self.render_mode = render_mode

        self.config = config

        self.viewer = Viewer(config["dt"])
        
        self.animal_count = config["animal"]["env"]["count"]
        self.animals = [Animal(config=config, 
                               behaviors=Behavior, 
                               movement_dims=MovementDim)
                        for _ in range(self.animal_count)]
        
        self.drone_count = config["drone"]["env"]["count"]
        self.drones = [Drone(config=config) 
                       for _ in range(self.drone_count)]
        
        self._env_steps = 0

    def _init_animal(self):
        # randomization using animal.rng, animal seed decides spawn location, spawn heading and behaviour
        for i, animal in enumerate(self.animals):
            animal.vel_dir = Vector(random_unit_2d=~animal.use_random_unit_3d,
                                    random_unit_3d=animal.use_random_unit_3d,
                                    rng=self.env_rng)
            animal.vel_speed = random.uniform(animal.min_speed, animal.max_speed)
            spawn_dir = Vector(random_unit_2d=~animal.use_random_unit_3d,
                               random_unit_3d=animal.use_random_unit_3d,
                               rng=self.env_rng)
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
            vel_dir = Vector(random_unit_3d=True, rng=self.sample_action_rng)
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

    def reset(self, seed=None):
        """
        Reinitializes drone and animal initial positions according to config.
        Returns observations, the animals and drones.
        
        :param seed: makes every episode the same
        """

        self.set_seed(seed)

        self._init_animal()
        self._init_drone()

        observations = self.in_FoV(self.drones, self.animals)

        info = {}
        return observations, info
    
    def package_actions(self, actions):

        n_actions = self.config.model.space.n_actions
        n_drones = self.config.drone.env.count

        packaged_actions = []

        for i in range(n_drones):
            start = i * n_actions
            end = start + n_actions
            drone_actions = actions[start:end]

            package_action = {
                "vel_dir": Vector(drone_actions[0],
                                drone_actions[1],
                                drone_actions[2]),
                "vel_speed": drone_actions[3],
                "theta": drone_actions[4]
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
            # movement
            animal.update_vel(rng=self.env_rng)
            animal.enforce_speed()
            animal.update_pos()
            animal.enforce_position()

    def in_FoV(self, drones, animals):
            """
            x: view dir
            z: perpendicular upwards relative to x-dir
            y: perpendicular right relative to x-dir
            """
            in_FoV_obs = []
            world_z = np.array([0,0,1], dtype=np.float32)

            for drone in drones:

                drone_obs = []

                # get camera / frustum view direction
                x = drone.view_dir.to_numpy()
                x = x / np.linalg.norm(x)

                # create horizontal basis
                y = np.cross(world_z, x)
                y = y / np.linalg.norm(y)

                # create upward basis
                z = np.cross(x, y)
                z = z / np.linalg.norm(z)

                # specified config values are full width
                v_max = np.deg2rad(drone.ver_angle / 2)
                h_max = np.deg2rad(drone.hor_angle / 2)

                for animal in animals:

                    # get animal and drone positions
                    drone_pos = drone.pos.to_numpy()
                    animal_pos = animal.pos.to_numpy()

                    # get vector pointing in the direction of the animal
                    rel_vec = animal_pos - drone_pos
                    distance = np.linalg.norm(rel_vec)

                    # guard against zero distance
                    drone_to_animal_vec = rel_vec / distance if distance >= 1e-8 else np.zeros(3)

                    cx = np.dot(drone_to_animal_vec, x)
                    cy = np.dot(drone_to_animal_vec, y)
                    cz = np.dot(drone_to_animal_vec, z)

                    # horizontal and vertical angles
                    v_angle = np.arctan2(cz, cx)
                    h_angle = np.arctan2(cy, cx)

                    # check if animal is in view
                    in_view = (
                        cx > 0 and 
                        abs(v_angle) <= v_max and
                        abs(h_angle) <= h_max and
                        distance <= drone.view_range
                    )

                    if in_view:

                        # normalize values for the observation
                        dist_norm = distance / drone.view_range
                        v_norm = v_angle / v_max
                        h_norm = h_angle / h_max

                        drone_obs.append([
                            1.0, 
                            dist_norm, 
                            v_norm, 
                            h_norm
                        ])

                    else:

                        # default values when not in view
                        drone_obs.append([
                            0.0, 
                            1.0, 
                            0.0, 
                            0.0
                        ])

                # flatten the observations for this specific drone
                in_FoV_obs.append(np.array(drone_obs, dtype=np.float32))

            return np.array(in_FoV_obs, dtype=np.float32)

    def compute_reward(self, observations):

        r_vis = 0.0
        r_dist = 0.0
        r_align = 0.0

        for d in range(self.drone_count):

            drone_obs = observations[d]

            in_view = drone_obs[:, 0]
            dist = drone_obs[:, 1]
            v = np.abs(drone_obs[:, 2])
            h = np.abs(drone_obs[:, 3])

            visible = in_view == 1.

            if np.any(visible):
                r_vis += np.mean(in_view)
                r_dist += np.mean(1.0 - dist[visible])
                r_align += np.mean(1.0 - (v[visible] + h[visible]) * 0.5)

        r_vis /= self.drone_count
        r_dist /= self.drone_count
        r_align /= self.drone_count

        rewards = [r_vis, r_dist, r_align]
        reward = sum(rewards) / len(rewards)

        return reward

    def set_render_mode(self, mode):
        self.render_mode = mode

    def render(self):
        self.viewer.draw(self.drones, self.animals, self.render_mode)

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

    def step(self, actions):
        self._env_steps += 1

        actions = self.package_actions(actions)
        self._step_drone(self.drones, actions)
        self._step_animal()

        reward = 0

        """
        Lost tracking → failure
        Animal stress > threshold → failure
        """
        terminated = False
        truncated = False
        info = {}

        if self._env_steps >= 128:
            self._env_steps = 0
            terminated = True

        """
        observations.shape =
            (
                num_drones,
                num_animals * [in_view, dist_norm, h_norm, v_norm]
            )
        """
        observations = self.in_FoV(self.drones, self.animals)

        reward = self.compute_reward(observations)

        self.render()
        return observations, reward, terminated, truncated, info