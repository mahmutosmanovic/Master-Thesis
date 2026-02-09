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

def random_direction(rng):
    return unit(np.array([
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
    ]))

def random_position(rng, x_max, y_max, z_max):
    return np.array([
        rng.uniform(0, x_max),
        rng.uniform(0, y_max),
        rng.uniform(0, z_max),
    ])

def position_on_cylinder(center, rng, xy_radius=120, z=60):
    angle = rng.uniform(0, 2*np.pi)
    offset = np.array([xy_radius * np.cos(angle), xy_radius * np.sin(angle), z], dtype=float)
    pos = center + offset
    xy_yaw = np.arctan2(center[1] - pos[1], center[0] - pos[0])

    return pos, xy_yaw

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