from enum import Enum

class Behavior(Enum):
    RANDOM = 5003
    POI = 5009
    PATH = 5011

class BehaviorState(Enum):
    EXPLORE = 83
    EXPLOIT = 89

class MovementDim(Enum):
    TWO_D = 43
    THREE_D = 47