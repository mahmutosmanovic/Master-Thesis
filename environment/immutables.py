from enum import Enum

class Behavior(Enum):
    RANDOM = 5003
    POI = 5009
    PATH = 5011

class MovementDim(Enum):
    TWO_D = 43
    THREE_D = 47