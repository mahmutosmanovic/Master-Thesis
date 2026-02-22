import random
import numpy as np
from .vec import Vector
from .immutables import Drone_Type
from .behaviors import BEHAVIOR_FNs
    
class Entity:
    next_id = 1
    def __init__(self, config):
        self.id = int(Entity.next_id)
        Entity.next_id += 1

        self.dt = config["dt"]

    def update_pos(self):
        self.pos.add(self.vel_dir.scale(self.vel_speed*self.dt), in_place=True)

    def enforce_speed(self):
        if self.vel_speed > self.max_speed:
            self.vel_speed = self.max_speed
        elif self.vel_speed < self.min_speed:
            self.vel_speed = self.min_speed
        
class Drone(Entity):
    def __init__(self, config, d_type="small"):
        super().__init__(config)
        # type
        self.drone_type = Drone_Type.SMALL if d_type == "small" else Drone_Type.LARGE 

        # camera
        self.ver_angle = int(config["drone"][d_type]["ver_angle"])
        self.hor_angle = int(config["drone"][d_type]["hor_angle"])
        self.disturbance_mult = config["drone"][d_type]["disturbance_mult"]
        self.view_range = int(config["drone"][d_type]["view_range"])
        self.max_cam_rot = int(config["drone"][d_type]["max_cam_rot"])
        self.max_altitude = int(config["drone"][d_type]["max_altitude"])
        self.view_dir = Vector(*config["drone"][d_type]["view_dir"])
        self.theta = 0 # camera rotation in degrees per timestep

        # pos
        self.spawn_dist = int(random.uniform(*config["drone"][d_type]["spawn_dist"]))
        self.pos = Vector()

        # movement
        self.min_speed = int(config["drone"][d_type]["min_speed"])
        self.max_speed = int(config["drone"][d_type]["max_speed"])
        self.vel_speed = 0
        self.vel_dir = Vector()
        
    def rotate_view(self):
        self.view_dir.rotate_z(self.theta)

    def reset_theta(self):
        self.theta = 0
    
    def enforce_cam_rot(self): 
        scaled_theta = self.max_cam_rot * self.dt
        if self.theta > scaled_theta:
            self.theta = scaled_theta
        elif self.theta < -scaled_theta:
            self.theta = -scaled_theta

    def enforce_position(self):
        if self.pos.z < 0:
            self.pos.z = 0

class Animal(Entity):
    def __init__(self, config, behaviors, movement_dims):
        super().__init__(config)
        # enums
        self.behaviors = behaviors
        self.movement_dims = movement_dims

        # movement
        self.min_speed = float(config["animal"]["init"]["min_speed"])
        self.max_speed = float(config["animal"]["init"]["max_speed"])
        self.epsilon = float(config["animal"]["init"]["epsilon"])
        self.ver_dir_angle = int(config["animal"]["init"]["ver_dir_angle"])
        self.hor_dir_angle = int(config["animal"]["init"]["hor_dir_angle"])
        self.behavior = config["animal"]["init"]["behavior"]
        self.movement_dim = config["animal"]["init"]["movement_dim"]
        self.use_random_unit_3d = (self.movement_dim == self.movement_dims.THREE_D)
        
        self.disturbance = 0
        self.escape_vector = np.zeros(3)

        self.vel_speed = 0
        self.vel_dir = Vector()

    def _enforce_position_3d(self):
        if self.pos.z < 0:
            self.pos.z = 0

    def _enforce_position_2d(self):
        if self.pos.z != 0:
            self.pos.z = 0 

    def enforce_position(self):
        if self.use_random_unit_3d:
            self._enforce_position_3d()
        elif ~self.use_random_unit_3d:
            self._enforce_position_2d()
        else:
            raise ValueError(f"Unexpected value: Choose 2D or 3D")

    def update_vel(self, rng):
        """
        Updates velocity (magnitude and direction) according to specified behavior (random walk, points of interest, path following) and movement dimensions (2D or 3D).
        """
        fn = BEHAVIOR_FNs[self.behavior]
        fn(self, rng)