from learning.robot_policy import RobotPolicy

class UGVPolicy(RobotPolicy):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        # returns [steering, throttle]
        pass
