from ..agent import Agent
import numpy as np

class Animal(Agent):
    def __init__(self, pos, mode="random"):
        super().__init__(pos)
        self.mode = mode

    def policy(self, obs):
        if self.mode == "random":
            return self.random_policy(obs)

        elif self.mode == "path_follow":
            return self.path_follow_policy(obs)

        elif self.mode == "poi":
            return self.static_poi_policy(obs)

        elif self.mode == "learning_based":
            return self.model_based_policy(obs)
        
    def random_policy(self, obs):
        turn_angle = np.random.normal(0.0, self.turn_noise)
        accel = np.random.normal(0.0, 5.0)
        return turn_angle, accel

class Eagle(Animal):
    def __init__(self, pos, mode="random"):
        super().__init__(pos, mode)
        # movement limits
        self.max_speed = 30.0      # m/s   (fast flight)
        self.max_turn  = 8.0       # rad/s (smooth, wide turns)
        self.max_accel = 8.0       # m/s^2 (powerful takeoff)

        # perception & cognition
        self.vision = 250.0        # m     (long-range vision)
        self.turn_noise = 2.5     # rad/s (very stable heading)
        self.memory_decay = 0.995  # unitless (slow forgetting)
        
    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"

class Jackal(Animal):
    def __init__(self, pos, mode="random"):
        super().__init__(pos, mode)

        # movement limits
        self.max_speed = 12.0      # m/s   (running speed)
        self.max_turn  = 4.0      # rad/s (body inertia)
        self.max_accel = 4.0       # m/s^2 (ground acceleration)

        # perception & cognition 
        self.vision = 100.0        # m     (horizontal vision)
        self.turn_noise = 1.5     # rad/s (moderate variability)
        self.memory_decay = 0.98   # unitless (good spatial memory)

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"

class Pigeon(Animal):
    def __init__(self, pos):
        super().__init__(pos)

        # movement limits
        self.max_speed = 15.0      # m/s   (typical cruising speed)
        self.max_turn  = 16.0       # rad/s (high maneuverability)
        self.max_accel = 6.0       # m/s^2 (quick bursts)

        # perception & cognition
        self.vision = 80.0         # m     (local perception)
        self.turn_noise = 3.25     # rad/s (exploratory jitter)
        self.memory_decay = 0.97   # unitless (faster forgetting)

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"