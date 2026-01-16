import numpy as np

class Agent:
    _next_id = 0

    def __init__(self, pos):
        self.agent_id = Agent._next_id
        Agent._next_id += 1

        self.pos = np.array(pos) # 3d
        self.speed = 0.0
        self.direction = self.random_direction()

    def l2_norm(self, v): # unit length
        return v / np.linalg.norm(v)

    def random_direction(self):
        v = np.random.normal(size=3)
        return self.l2_norm(v)

    def move(self, dt):
        self.pos += self.direction * self.speed * dt

    def update(self, dt, observation):
        angular_velocity, accel = self.policy(observation)
        self.apply_control(angular_velocity, accel, dt)
        self.move(dt)
        return angular_velocity, accel

    def apply_control(self, angular_velocity, accel, dt):
        angular_velocity = np.clip(
            angular_velocity, -self.max_turn, self.max_turn
        )
        accel = np.clip(accel, -self.max_accel, self.max_accel)

        angle = angular_velocity * dt
        c, s = np.cos(angle), np.sin(angle)

        x, y, _ = self.direction
        self.direction = np.array([c*x - s*y, s*x + c*y, 0.0])
        self.direction /= np.linalg.norm(self.direction)

        self.speed = np.clip(
            self.speed + accel * dt, 0, self.max_speed
        )


