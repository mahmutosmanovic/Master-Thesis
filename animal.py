from settings import *

class Animal:
    def __init__(self, config, behavior='random', start_pos=(0, 0, 0)):
        self.behavior = behavior
        self.config = config

        self.x, self.y, self.z = start_pos
        self.z = 0

        self.v = np.zeros(2)

    def reset(self, start_pos=(0,0,0)):

        self.x, self.y, self.z = start_pos
        self.z = 0.0

        self.v = np.zeros(2)


    def step(self):
        """
        Perform one simulation step.
        Returns (x, y, z)
        """

        if self.behavior == "random":
            self._random_behavior()

        elif self.behavior == "path":
            self._path_behavior()

        elif self.behavior == "poi":
            self._poi_behavior()

        elif self.behavior == "learn":
            self._learn_behavior()

        else:
            raise ValueError(f"Unknown behavior: {self.behavior}")

        # Enforce planar constraint
        self.z = 0.0

        return self.x, self.y, self.z


    def _random_behavior(self):
            # 1. Generate a random target direction
            rand_dir = np.random.normal(0, 1, 2)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-9)

            # 2. Convert that direction to a desired step
            target_v = rand_dir * self.config.max_speed

            # 3. Interpolate from current velocity to target (Inertia)
            # epsilon = 0.1 means 10% new direction, 90% old direction
            self.v = (1 - self.config.epsilon) * self.v + self.config.epsilon * target_v

            # 4. Final speed clamp (Safety check)
            speed = np.linalg.norm(self.v)
            if speed > self.config.max_speed:
                self.v *= self.config.max_speed / speed

            # 5. Update position
            self.x += self.v[0]
            self.y += self.v[1]


    def _path_behavior(self):
        print("Path mode, speed =", self.config.max_speed)


    def _poi_behavior(self):
        print("POI mode, range =", self.config.vision_range)


    def _learn_behavior(self):
        print("Learning rate =", self.config.learning_rate)