import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
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
    yaw_speed = 0.50
    max_speed: float = 15.0
    fov: float = np.deg2rad(90)
    learning_rate: float = 0.02

STEPS = 500
EPISODES = 200
DATA_FOLDER_PATH = "data/pigeon/animal_01.csv"
CSV_PATH = "log.csv"
BEHAVIOR = "random"



