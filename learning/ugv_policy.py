from learning.robot_policy import RobotPolicy

class UAVPolicy(RobotPolicy):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        # returns [roll, pitch, yaw, throttle]
        pass
