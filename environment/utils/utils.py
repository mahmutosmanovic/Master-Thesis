import json
import numpy as np
from .vec_utils import unit
from dataclasses import asdict, is_dataclass

def log_config_text(writer, cfg, tag="config/env"):
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = cfg.__dict__
    else:
        payload = cfg

    txt = json.dumps(payload, indent=2)
    writer.add_text(tag, f"```json\n{txt}\n```")

def decode_action(a: np.ndarray):
    a = np.asarray(a, dtype=np.float32)

    direction = unit(a[:3])

    # speed: [0,1]
    speed = float((a[3] + 1.0) * 0.5)

    # yaw rate: [-1,1]
    view_yaw_rate = float(a[4])

    return (direction, speed, view_yaw_rate)