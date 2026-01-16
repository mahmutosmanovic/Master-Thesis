import argparse

from experiments.train_animals import train_animals
from experiments.train_robots import train_robots
from view.settings import run_viewer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["train_animals", "train_robots", "view"],
                        required=True)

    args = parser.parse_args()

    if args.mode == "train_animals":
        train_animals()

    elif args.mode == "train_robots":
        train_robots()

    elif args.mode == "view":
        run_viewer()

if __name__ == "__main__":
    main()
