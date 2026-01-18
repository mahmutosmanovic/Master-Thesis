from ..agent import Agent
import numpy as np


class Animal(Agent):
    def __init__(self, pos, mode="random"):
        super().__init__(pos)
        self.mode = mode

        # path following
        self.path = None
        self.s = 0.0               # path parameter
        self.s_speed = 0.5         # how fast we move along the path
        self.epsilon = 0.05        # deviation probability

    # Policy dispatch
    def policy(self, obs):
        if self.mode == "random":
            return self.random_policy(obs)

        elif self.mode == "path_follow":
            return self.path_follow_policy(obs)

        elif self.mode == "poi":
            return self.static_poi_policy(obs)

        elif self.mode == "learning_based":
            return self.model_based_policy(obs)

        # fallback
        return self.random_policy(obs)

    # Policies
    def random_policy(self, obs):
        turn_angle = np.random.normal(0.0, self.turn_noise)
        accel = np.random.normal(0.0, 5.0)
        return turn_angle, accel

    def path_follow_policy(self, obs):
        if self.path is None:
            return self.random_policy(obs)

        pos = obs["pos"]
        direction = obs["direction"]

        center = self.path.center
        radius = self.path.radius

        # vector from center to agent
        rel = pos - center
        rel[2] = 0.0
        dist = np.linalg.norm(rel)

        if dist < 1e-6:
            return 0.0, 0.0

        # closest point on the circle
        radial_dir = rel / dist
        closest_point = center + radius * radial_dir

        # tangent direction (CCW)
        tangent = np.array([
            -radial_dir[1],
            radial_dir[0],
            0.0
        ])

        # path attraction
        to_path = closest_point - pos
        Kp = 1.0

        desired = tangent + Kp * to_path
        desired /= np.linalg.norm(desired)

        # heading error
        cross = direction[0] * desired[1] - direction[1] * desired[0]
        dot = np.dot(direction[:2], desired[:2])
        turn_angle = np.arctan2(cross, dot)

        # mild epsilon disturbance
        if np.random.rand() < self.epsilon:
            turn_angle += np.random.normal(0.0, self.turn_noise * 0.3)

        desired_speed = 0.7 * self.max_speed
        accel = desired_speed - obs["speed"]

        return turn_angle, accel


# Species
class Eagle(Animal):
    def __init__(self, pos, mode="random"):
        super().__init__(pos, mode)
        
        # path following
        self.epsilon = 0.3

        # movement limits
        self.max_speed = 30.0
        self.max_turn  = 8.0
        self.max_accel = 8.0

        # perception & cognition
        self.vision = 250.0
        self.turn_noise = 2.5
        self.memory_decay = 0.995

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"


class Jackal(Animal):
    def __init__(self, pos, mode="random"):
        super().__init__(pos, mode)

        # path following
        self.epsilon = 0.6

        # movement limits
        self.max_speed = 6.0
        self.max_turn  = 4.0
        self.max_accel = 4.0

        # perception & cognition
        self.vision = 100.0
        self.turn_noise = 1.5
        self.memory_decay = 0.98

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"


class Pigeon(Animal):
    def __init__(self, pos, mode="random"):
        super().__init__(pos, mode)
        
        # path following
        self.epsilon = 0.8

        # movement limits
        self.max_speed = 15.0
        self.max_turn  = 16.0
        self.max_accel = 6.0

        # perception & cognition
        self.vision = 80.0
        self.turn_noise = 3.25
        self.memory_decay = 0.97

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={self.mode})"
