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
        actions = []
        for drone in self.drones:
            action = {
                "vel_dir": Vector(random_unit_3d=True),
                "vel_speed": random.uniform(drone.min_speed, drone.max_speed),
                "theta": random.triangular(-drone.max_cam_rot, drone.max_cam_rot)
            }
            actions.append(action)
        return actions
        
    def reset(self, seed=None):
        if seed:
            random.seed(seed)

        self._init_animal()
        self._init_drone()

        info = {}

        return (self.animals, self.drones), info
    
    def step(self, actions):
        for drone, action in zip(self.drones, actions):
            drone.vel_dir.setter(action["vel_dir"])
            drone.vel_speed = action["vel_speed"]
            drone.theta = action["theta"]
            drone.update_pos()

    def close(self):
        ...
        