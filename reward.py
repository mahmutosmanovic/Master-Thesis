from settings import *
from disturbance import DisturbanceField

_df = DisturbanceField()

def compute_total_reward(next_obs, animal_pos, drone_pos):
    """
    next_obs: (in_view, ang_score, range_score) from drone.observe(pigeon)
    animal_pos: (x,y,z)
    drone_pos: (x,y,z)
    returns: total_reward, monitoring_reward, disturbance, disturbance_penalty
    """

    in_view, ang_score, range_score = next_obs

    # Your existing monitoring term (kept exactly)
    monitoring_reward = float(in_view) * float(ang_score) * float(range_score)

    # Disturbance in [0,1]
    disturbance = _df.get_disturbance(animal_pos, drone_pos)

    # Hinge penalty only above a threshold, normalized back to [0,1]
    excess = max(0.0, disturbance - DISTURB_THRESHOLD)
    if (1.0 - DISTURB_THRESHOLD) > 1e-8:
        excess_norm = excess / (1.0 - DISTURB_THRESHOLD)
    else:
        excess_norm = 0.0

    disturbance_penalty = excess_norm ** float(DISTURB_POWER)

    total_reward = MONITOR_W * monitoring_reward - DISTURB_W * disturbance_penalty

    return total_reward, monitoring_reward, disturbance_penalty
