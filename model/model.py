import numpy as np

class Model:
    def __init__(self):
        ...

    def policy(self, observation):
        # direction = (np.random.random(3) - 0.5)*2
        direction = np.array([1,0,0])
        n = np.linalg.norm(direction)
        if n < 1e-8:
            direction /= np.linalg.norm(direction)
        else:
            direction = np.zeros(3, dtype=float)

        # speed = (np.random.random() - 0.5)*2
        speed = float(1)
        camera_yaw = float(1)

        return direction, speed, camera_yaw