import numpy as np
from environment.agents.sensor import SensorMetrics

def tracking_reward(
    sensor_metrics: SensorMetrics,
    disturbance: float,
    distance_scale=5.0,
    alignment_scale=1.0,
    disturbance_scale=1.0,
):
    if sensor_metrics.n_visible == 0:
        return 0

    r_distance = distance_scale * (1.0 - sensor_metrics.mean_distance)
    r_alignment = -alignment_scale * sensor_metrics.alignment_error
    r_disturbance = -disturbance_scale * disturbance

    return (r_distance + r_alignment) * sensor_metrics.n_visible + r_disturbance