import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return v / n

def angle_between(a, b):
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)

def slerp(a, b, t): # If we want to enforce max_turn
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 0.9995:
        return unit((1 - t) * a + t * b)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    return (
        np.sin((1 - t) * theta) / sin_theta * a +
        np.sin(t * theta) / sin_theta * b
    )