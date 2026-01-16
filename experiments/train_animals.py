from settings import *
from data.gps import load_tracks # type: ignore
from simulation.world import World

def train_animals():
    gps_tracks = load_tracks()

    for track in gps_tracks:
        world = World.from_track(track)

        for episode in range(EPISODES_TRAIN_ANIMALS):
            world.reset()
            while not world.done:
                world.step(TIME_STEP)

            world.learn_from_track(track)
