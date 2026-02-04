from settings import *

def compute_total_reward(next_obs, animal_pos=None, drone_pos=None, prev_obs=None):
    in_view, ang_score, range_score = next_obs
    eps = 1e-6

    # Soft visibility
    vis = in_view * (0.4 + 0.6 * ang_score)

    # State reward (bounded, smooth)
    center_error = 1.0 - ang_score
    center_term = np.exp(-(center_error ** 2) / CENTER_SIGMA)

    range_term = np.exp(-((range_score - RANGE_TARGET) ** 2) / RANGE_SIGMA)

    state_reward = VIEW_W * vis * center_term * range_term

    # --- ERROR REDUCTION REWARD (CRITICAL) ---
    correction_reward = 0.0
    if prev_obs is not None:
        _, prev_ang, prev_range = prev_obs
        correction_reward = CORRECT_W * (
            (prev_ang - ang_score)   # positive if centering improved
            + (range_score - prev_range)
        )

    # Smooth close repulsion
    close_penalty = 0.0
    if animal_pos is not None and drone_pos is not None:
        ax, ay, az = animal_pos
        dx, dy, dz = drone_pos
        dist = np.sqrt((ax - dx)**2 + (ay - dy)**2 + (az - dz)**2 + eps)
        close_penalty = -CLOSE_W * np.exp(-(dist / CLOSE_RADIUS) ** 2)

    total_reward = state_reward + correction_reward + close_penalty
    return total_reward, state_reward, close_penalty
