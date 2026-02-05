from settings import *
from disturbance import *

DIST_FIELD = DisturbanceField()

def compute_total_reward(next_obs, animal_pos, drone_pos, prev_obs=None):

    relx, rely, relz, dist_norm, fov_margin = next_obs

    # --------------------
    # Tracking reward
    # --------------------

    k = 10.0
    in_fov = 1.0 / (1.0 + np.exp(-k * fov_margin))

    close = np.clip((0.5 - dist_norm) / 0.5, 0.0, 1.0)

    monitoring_r = in_fov + 0.8 * in_fov * close

    recover_pen = -0.2 * (1.0 - in_fov)

    # --------------------
    # Disturbance penalty
    # --------------------

    disturbance = DIST_FIELD.get_disturbance(animal_pos, drone_pos)

    eco_pen = -DIST_W * disturbance

    # --------------------
    # Total
    # --------------------

    total = monitoring_r + recover_pen + eco_pen

    return total, monitoring_r, eco_pen