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

        # cant move underground
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0

    def update(self, dt, observation):
        yaw_rate, pitch_rate, accel = self.policy(observation, dt)
        self.apply_control(yaw_rate, pitch_rate, accel, dt)
        self.move(dt)
        return yaw_rate, pitch_rate, accel

    def apply_control(self, yaw_rate, pitch_rate, accel, dt):
        # Clip controls
        yaw_rate   = np.clip(yaw_rate,   -self.max_turn, self.max_turn)
        pitch_rate = np.clip(pitch_rate, -self.max_turn, self.max_turn)
        accel      = np.clip(accel,      -self.max_accel, self.max_accel)

        d = self.direction

        # Planar constraint
        if self.is_planar:
            pitch_rate = 0.0
            d = np.array([d[0], d[1], 0.0])
            d /= np.linalg.norm(d) + 1e-12

        # Local frame (no roll)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(d, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= np.linalg.norm(right)

        up = np.cross(right, d)

        omega = yaw_rate * up + pitch_rate * right
        ang = np.linalg.norm(omega)

        if ang > 1e-8:
            axis = omega / ang
            theta = ang * dt
            d = (
                d * np.cos(theta)
                + np.cross(axis, d) * np.sin(theta)
                + axis * np.dot(axis, d) * (1 - np.cos(theta))
            )

        self.direction = d / (np.linalg.norm(d) + 1e-12)
        self.speed = np.clip(self.speed + accel * dt, 0.0, self.max_speed)
