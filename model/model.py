import numpy as np

class Model:
    def __init__(self):
        ...

    def policy(self, observation):
        yaw = (np.random.random() - 0.5)*2
        pitch = 0
        accel = (np.random.random() - 0.5)*2

        return yaw, pitch, accel