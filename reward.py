from settings import *
from disturbance import *

DIST_FIELD = DisturbanceField()

def compute_total_reward(next_obs, animal_pos=None, drone_pos=None, prev_obs=None):

    relx, rely, relz, dist_norm, fov_margin = next_obs

    # ----------------------------
    # FoV quality (0..1)
    # ----------------------------

    k = 10.0
    in_fov = 1.0 / (1.0 + np.exp(-k * fov_margin))


    # ----------------------------
    # Distance quality (0..1)
    # Target: ~50% of max range
    # ----------------------------

    TARGET = 0.5      # ideal distance (normalized)
    SIGMA = 0.15      # tolerance

    dist_quality = np.exp(-((dist_norm - TARGET) ** 2) / (2 * SIGMA ** 2))


    # ----------------------------
    # Main monitoring (dominant)
    # ----------------------------
    # Must have both good FoV AND good distance

    monitoring_r = 2.0 * in_fov * dist_quality


    # ----------------------------
    # Recovery penalty
    # ----------------------------

    recover_pen = -0.3 * (1.0 - in_fov)


    # ----------------------------
    # Safety barrier (too close)
    # ----------------------------

    MIN_DIST = 0.25   # 25% of range

    close_pen = -3.0 * max(0.0, MIN_DIST - dist_norm)


    # ----------------------------
    # Total
    # ----------------------------

    total = monitoring_r + recover_pen + close_pen

    return total, monitoring_r, recover_pen + close_pen
