import os
import math
import torch
import neptune
import numpy as np
import pandas as pd
import torch.nn as nn
import geopandas as gpd
import torch.optim as optim
from collections import deque
from matplotlib import colors
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from dataclasses import dataclass
from shapely.geometry import Point
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

load_dotenv()

@dataclass
class AnimalConfig:
    epsilon: float = 0.1
    step_size: float = 1.0
    max_speed: float = 20.0

FOV_DEG = 90
MAX_VIEW_RANGE = 200
@dataclass
class DroneConfig:
    yaw_speed = 0.30
    max_speed: float = 15.0
    fov: float = np.deg2rad(FOV_DEG)
    max_view_range: int = MAX_VIEW_RANGE


STEPS = 500
EPISODES = 500
ROLLOUT_EPS = 3
learning_rate = 0.0005
obs_dim = 5 # obs = (in_view, angle, dist) -> 3
act_dim = 4 # action = dx,dy,dz,dyaw -> 4
BEHAVIOR = "random"
DATA_FOLDER_PATH = "data/pigeon/animal_01.csv"
CSV_PATH = "log.csv"


# action limimts
MAX_DX = 4.0
MAX_DY = 4.0
MAX_DZ = 1.0
MAX_DYAW = 1.0



