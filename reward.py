from settings import *
from disturbance import DisturbanceField

_df = DisturbanceField()

def compute_total_reward(next_obs, animal_pos, drone_pos):
    """
    Returns:
      total_reward, monitoring_reward, disturbance, disturbance_penalty
    """

    in_view, ang_score, range_score = next_obs

    # -----------------------------
    # Monitoring reward (unchanged)
    # -----------------------------
    monitoring_reward = (
        float(in_view)
        + float(ang_score)
        + float(range_score)
    )

    # -----------------------------
    # Disturbance (0..1)
    # -----------------------------
    disturbance = _df.get_disturbance(animal_pos, drone_pos)

    # -----------------------------
    # Smooth band-pass penalty
    # -----------------------------
    # Preferred disturbance level (sweet spot)
    D_TARGET = DISTURB_THRESHOLD        # reuse your existing setting
    D_SIGMA  = 0.25                     # width of acceptable band

    disturbance_penalty = ((disturbance - D_TARGET) / D_SIGMA) ** 2

    disturbance_penalty = 0
    # -----------------------------
    # Total reward
    # -----------------------------
    total_reward = (
        MONITOR_W * monitoring_reward
        - DISTURB_W * disturbance_penalty
    )

    return total_reward, monitoring_reward, disturbance_penalty
