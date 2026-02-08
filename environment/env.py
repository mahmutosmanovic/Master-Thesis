import random
from .vec import Vector
from .entity import Drone, Animal
from scripts import Behavior, MovementDim

class Env:
    def __init__(self, config, render_mode=None, seed=None):
        if seed:
            random.seed(seed)

        self.render_mode = render_mode

        self.config = config
        
        self.animal_count = config["animal"]["env"]["count"]
        self.animals = [Animal(config=config, 
                               behaviors=Behavior, 
                               movement_dims=MovementDim)
                        for _ in range(self.animal_count)]
        
        self.drone_count = config["drone"]["env"]["count"]
        self.drones = [Drone(config=config) 
                       for _ in range(self.drone_count)]

    def _init_animal(self):
        for animal in self.animals:
            animal.vel_dir = Vector(random_unit_2d=~animal.use_random_unit_3d,
                                    random_unit_3d=animal.use_random_unit_3d)
            animal.vel_speed = random.uniform(animal.min_speed, animal.max_speed)
            spawn_dir = Vector(random_unit_2d=~animal.use_random_unit_3d,
                               random_unit_3d=animal.use_random_unit_3d)
            radis_len = random.uniform(0, self.config["animal"]["init"]["max_spawn_radius"])
            animal.pos = spawn_dir.scale(radis_len)

    def _init_drone(self):
        for drone, animal in zip(self.drones, self.animals):
            animal_pos = animal.pos.getter()    
            drone.view_dir.unit()
            drone.view_dir.rotate_z(random.uniform(0,360))
            view_dir = drone.view_dir.getter()
            inv_scale_view_dir = view_dir.scale(-drone.spawn_dist)
            new_drone_pos = animal_pos.add(inv_scale_view_dir)
            drone.pos.setter(new_drone_pos)
            drone.vel_speed = random.uniform(drone.min_speed, drone.max_speed)

    def sample_action(self):
        """
        Action Space:
            1. Velocity Direction (unit vector): [vx,vy,vz],
            2. Velocity speed (between min-max in config): k,
            3. Angle, the amount of degrees to rotate the camera (view_dir, between -max_cam_rot and max_cam_rot): theta.
        """

        actions = []
        for drone in self.drones:
            action = {
                "vel_dir": Vector(random_unit_3d=True),
                "vel_speed": random.uniform(drone.min_speed, drone.max_speed),
                "theta": random.triangular(-drone.max_cam_rot, 0, drone.max_cam_rot)
            }
            actions.append(action)
        return actions
        
    def reset(self, seed=None):
        """
        Reinitializes drone and animal initial positions according to config.
        Returns observations, the animals and drones.
        
        :param seed: makes every episode the same
        """

        if seed:
            random.seed(seed)

        self._init_animal()
        self._init_drone()

        info = {}

        return (self.animals, self.drones), info
    
    def _step_drone(self, drones, actions):
        for drone, action in zip(drones, actions):
            # movement
            drone.vel_dir.setter(action["vel_dir"])
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
            animal.update_vel()
            animal.enforce_speed()
            animal.update_pos()
            animal.enforce_position()
    
    def step(self, actions):
        self._step_drone(self.drones, actions)
        self._step_animal()

        reward = 0

        terminated = False
        truncated = False
        info = {}

        observations = (self.animals, self.drones)
        return observations, reward, terminated, truncated, info

    def close(self):
        ...
        