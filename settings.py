import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import geopandas as gpd
import torch.optim as optim
from matplotlib import colors
import matplotlib.pyplot as plt
from dataclasses import dataclass
from shapely.geometry import Point
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

@dataclass
class PigeonConfig:
    epsilon: float = 0.1
    step_size: float = 1.0
    max_speed: float = 15.0

@dataclass
class DroneConfig:
    yaw_speed = 0.20
    max_speed: float = 15.0
    fov: float = np.deg2rad(90)

STEPS = 200
EPISODES = 500
obs_dim = 3 # obs = (in_view, angle, dist) -> 3
act_dim = 4 # action = dx,dy,dz,dyaw -> 4
learning_rate = 0.02
DATA_FOLDER_PATH = "data/pigeon/animal_01.csv"
CSV_PATH = "log.csv"
BEHAVIOR = "random"

MAX_DX = 2.0
MAX_DY = 2.0
MAX_DZ = 1.0
MAX_DYAW = 1.0

# Reward weights
MONITOR_W = 1.0
DISTURB_W = 1.0

# Species-specific "safe" disturbance threshold (0..1)
DISTURB_THRESHOLD = 0.25

# Penalty shaping: 1 = linear, 2 = quadratic (harsher near high disturbance)
DISTURB_POWER = 2.0




