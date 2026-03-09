import numpy as np
from .vec import Vector
from .immutables import MovementDim
from .behaviors import BEHAVIOR_REGISTRY
    
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
        self.drone_type = d_type

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
        self.spawn_dist = config["drone"][d_type]["spawn_dist"]
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
    def __init__(self, config):
        super().__init__(config)
        self.resource_map = None # set in init

        # movement
        self.min_speed = float(config["animal"]["init"]["min_speed"])
        self.max_speed = float(config["animal"]["init"]["max_speed"])

        behavior_cfg = config["animal"]["init"]["behavior"]
        behavior_cls = BEHAVIOR_REGISTRY[type(behavior_cfg)]
        self.behavior = behavior_cls(behavior_cfg)

        self.movement_dim = config["animal"]["init"]["movement_dim"]
        
        self.disturbance = 0
        self.state = "calm"
        self.escape_dir = np.zeros(3)

        self.vel_speed = 0
        self.vel_dir = Vector()

    def _enforce_position_3d(self):
        if self.pos.z < 0:
            self.pos.z = 0

    def _enforce_position_2d(self):
        if self.pos.z != 0:
            self.pos.z = 0 

    def enforce_position(self):
        if self.movement_dim == MovementDim.THREE_D:
            self._enforce_position_3d()
        elif self.movement_dim == MovementDim.TWO_D:
            self._enforce_position_2d()
        else:
            raise ValueError(f"Unexpected value: Choose 2D or 3D")

    def update_vel(self, rng):
        """
        Updates velocity (magnitude and direction) according to specified behavior (random walk, points of interest, path following) and movement dimensions (2D or 3D).
        """
        return self.behavior.fn(self, rng, self.dt)