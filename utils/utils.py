import numpy as np
from utils.vec_utils import unit

def decode_action(a: np.ndarray):
    a = np.asarray(a, dtype=np.float32)

    direction = unit(a[:3])

    # speed: [0,1]
    speed = float((a[3] + 1.0) * 0.5)

    # yaw rate: [-1,1]
    view_yaw_rate = float(a[4])

    return (direction, speed, view_yaw_rate)