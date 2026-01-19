import numpy as np

class Agent:
    _next_id = 0

    def __init__(self, pos, seed):
        self.agent_id = Agent._next_id
        Agent._next_id += 1

        self.rng = np.random.default_rng(seed)

        self.pos = np.array(pos) # 3d
        self.speed = 0.0
        self.direction = self.random_direction()

    def l2_norm(self, v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return v / n

    def random_direction(self):
        v = self.rng.normal(size=3)
        return self.l2_norm(v)

    def move(self, dt):
        self.pos += self.direction * self.speed * dt

    def update(self, dt, observation):
        angular_velocity, accel = self.policy(observation)
        self.apply_control(angular_velocity, accel, dt)
        self.move(dt)
        return angular_velocity, accel

    def apply_control(self, turn_angle, accel, dt):
        max_turn_angle = self.max_turn * dt
        turn_angle = np.clip(turn_angle, -max_turn_angle, max_turn_angle)

        accel = np.clip(accel, -self.max_accel, self.max_accel)

        c, s = np.cos(turn_angle), np.sin(turn_angle)
        x, y, _ = self.direction
        self.direction = np.array([c*x - s*y, s*x + c*y, 0.0])
        self.direction /= np.linalg.norm(self.direction)

        self.speed = np.clip(self.speed + accel * dt, 0, self.max_speed)

