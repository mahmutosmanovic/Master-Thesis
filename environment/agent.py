import random
from .vec import Vector
    
class Agent:
    next_id = 1
    def __init__(self, config):
        self.id = int(Agent.next_id)
        Agent.next_id += 1

        self.dt = config["dt"]

    def update_pos(self):
        self.pos.add(self.vel.scale(self.dt))

    def update_vel(self):
        raise NotImplementedError

    def enforce_constraints(self):
        raise NotImplementedError

    def _enforce_speed(self):
        if self.vel_speed > self.max_speed:
            self.vel_speed = self.max_speed
        elif self.vel_speed < self.min_speed:
            self.vel_speed = self.min_speed

class Drone(Agent):
    def __init__(self, config):
        super().__init__(config)
        # speed
        self.min_speed = int(config["drone"]["init"]["min_speed"])
        self.max_speed = int(config["drone"]["init"]["max_speed"])
        self.vel_speed = random.uniform(self.min_speed, self.max_speed)

        # camera
        self.ver_angle = int(config["drone"]["init"]["ver_angle"])
        self.hor_angle = int(config["drone"]["init"]["hor_angle"])
        self.max_cam_rot = int(config["drone"]["init"]["max_cam_rot"])
        self.view_range = int(config["drone"]["init"]["view_range"])
        self.view_dir = Vector(*config["drone"]["init"]["view_dir"])
        self.theta = 0 # camera rotation in degrees per timestep

        # pos
        self.pos = Vector()
        
    def rotate_view(self):
        self.view_dir.rotate_z(self.theta)

    def reset_theta(self):
        self.theta = 0
    
    def _enforce_cam_rot(self): 
        scaled_theta = self.max_cam_rot * self.dt
        if self.theta > scaled_theta:
            self.theta = scaled_theta
        elif self.theta < -scaled_theta:
            self.theta = -scaled_theta

    def _enforce_position(self):
        if self.pos.z < 0:
            self.pos.z = 0

    def enforce_constraints(self):
        self._enforce_cam_rot()
        self._enforce_speed()
        self._enforce_position()        
        
class Animal(Agent):
    def __init__(self, config, behaviors, movement_dims):
        super().__init__(config)
        # enums
        self.behaviors = behaviors
        self.movement_dims = movement_dims

        # movement
        self.min_speed = float(config["animal"]["init"]["min_speed"])
        self.max_speed = float(config["animal"]["init"]["max_speed"])
        self.epsilon = int(config["animal"]["init"]["epsilon"])
        self.ver_dir_angle = int(config["animal"]["init"]["ver_dir_angle"])
        self.hor_dir_angle = int(config["animal"]["init"]["hor_dir_angle"])
        self.behavior = config["animal"]["init"]["behavior"]
        self.movement_dim = config["animal"]["init"]["movement_dim"]
        self.use_random_unit_3d = (self.movement_dim == self.movement_dims.THREE_D)
        self.vel_speed = random.uniform(self.min_speed, self.max_speed)
        self.vel_dir = Vector(random_unit_2d=~self.use_random_unit_3d,
                              random_unit_3d=self.use_random_unit_3d)
        
        # pos
        spawn_dir = Vector(random_unit_2d=~self.use_random_unit_3d,
                           random_unit_3d=self.use_random_unit_3d)
        radis_len = random.uniform(0, config["animal"]["init"]["max_spawn_radius"])
        spawn_dir.scale(radis_len)
        self.pos = spawn_dir

    def _enforce_position_3d(self):
        if self.pos.z < 0:
            self.pos.z = 0

    def _enforce_position_2d(self):
        if self.pos.z != 0:
            self.pos.z = 0 

    def _enforce_position(self):
        if self.use_random_unit_3d:
            self._enforce_position_3d()
        elif ~self.use_random_unit_3d:
            self._enforce_position_2d()
        else:
            raise ValueError(f"Unexpected value: Choose 2D or 3D")

    def enforce_constraints(self):
        self._enforce_speed()
        self._enforce_position()
        
        
        
