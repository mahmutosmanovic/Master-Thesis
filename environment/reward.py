import numpy as np
from environment.agents.sensor import SensorMetrics

def tracking_reward(sensor_metrics: SensorMetrics, disturbance: float, action, cfg):
    direction, norm_speed, view_yaw_rate = action
    r_control = -cfg.control_scale * view_yaw_rate**2

    r_disturbance = -cfg.disturbance_scale * disturbance
    if sensor_metrics.n_visible == 0:
        return r_disturbance

    r_distance = cfg.distance_scale * (1.0 - sensor_metrics.mean_distance)
    r_alignment = -cfg.alignment_scale * sensor_metrics.alignment_error

    return r_distance + r_alignment + r_disturbance + r_control