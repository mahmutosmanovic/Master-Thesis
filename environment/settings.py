import os
import csv
import numpy as np
from random import randint, uniform, choice

# World settings
DT = 0.2
MAX_T = 1000
MAP_HEIGHT = 200
MAP_WIDTH = 200
MAP_ALTITUDE = 100

DIST_PENALTY_SCALE = 2.5
POS_SCALE = np.array([MAP_WIDTH, MAP_HEIGHT, MAP_ALTITUDE], dtype=np.float32)

JACKAL_COUNT = 1
JACKAL_MODE = 'random'

EAGLE_COUNT = 0
EAGLE_MODE = 'random'

PIGEON_COUNT = 0
PIGEON_MODE = 'path_follow'

# POI settings
POI_COUNT = 3                               # number of points if generating randomly
POI_POINTS = [(30,0,0), (0,0,0), (0,30,30)]  # explicit list like [(10,20,0), (80,60,0)] overrides POI_COUNT
 
POI_REACHED_EPS = 3.0                       # meters, considered "arrived"
POI_SWITCH_ON_REACH = True                  # if True, pick next target when reached

# Proportional control parameters
YAW_GAIN = 1.0
PITCH_GAIN = 1.0                         # higher -> turns more aggressively toward desired direction
ACCEL_GAIN = 1.0                        # higher -> speed matches desired speed faster
NOISE_SCALE = 0.5                       # scale turn

# Drones
DRONE_COUNT = 1

# RL AGENT
EPISODES_TRAIN_ROBOTS = 10_000
EPISODES_TRAIN_ANIMALS = 1_000