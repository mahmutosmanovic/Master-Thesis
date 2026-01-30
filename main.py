from settings import *
from utils import *
from view import *

@dataclass
class PigeonConfig:
    epsilon: float = 0.05
    max_speed: float = 10.0
    vision_range: float = 100.0
    learning_rate: float = 0.01


def main():
    df = pd.read_csv("data/pigeon/animal_01.csv")
    df = transform_to_epsg32636(df)
    draw_trail_3D(df, interval=150)


if __name__ == "__main__":
    main()
