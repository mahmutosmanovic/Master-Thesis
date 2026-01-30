class Pigeon:
    def __init__(self, config, behavior='random'):
        self.behavior = behavior
        self.config = config

    def run(self):

        if self.behavior == "random":
            self._random_behavior()

        elif self.behavior == "path":
            self._path_behavior()

        elif self.behavior == "poi":
            self._poi_behavior()

        elif self.behavior == "learn":
            self._learn_behavior()

        else:
            raise ValueError(f"Unknown behavior: {self.behavior}")
        

    def _random_behavior(self):
        print("Random with epsilon =", self.config.epsilon)


    def _path_behavior(self):
        print("Path mode, speed =", self.config.max_speed)


    def _poi_behavior(self):
        print("POI mode, range =", self.config.vision_range)


    def _learn_behavior(self):
        print("Learning rate =", self.config.learning_rate)