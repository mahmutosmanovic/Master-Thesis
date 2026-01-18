import os
import csv
import numpy as np
from random import randint, uniform, choice

# World settings
MAP_HEIGHT = 100
MAP_WIDTH = 100

JACKAL_COUNT = 1
JACKAL_MODE = 'path_follow'

EAGLE_COUNT = 0
EAGLE_MODE = 'random' # unused...

PIGEON_COUNT = 0
PIGEON_MODE = 'random'

# POI settings
POI_COUNT = 3                               # number of points if generating randomly
POI_POINTS = [(30,0,0), (0,0,0), (0,30,0)]  # explicit list like [(10,20,0), (80,60,0)] overrides POI_COUNT
 
POI_REACHED_EPS = 3.0                       # meters, considered "arrived"
POI_SWITCH_ON_REACH = True                  # if True, pick next target when reached

# POI Steering (simple proportional control)
POI_TURN_GAIN = 1.0                         # higher -> turns more aggressively toward POI
POI_ACCEL_GAIN = 1.0                        # higher -> speed matches desired speed faster
POI_NOISE_SCALE = 0.5                       # scale turn

