from environment.config import AnimalParams, DroneParams

# Animals (placeholders, should be changed to 3 general "animals")
def jackal_params():
    return AnimalParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        max_accel=4.0,
        turn_noise=0.4,
        epsilon=1,
    )

def eagle_params():
   return AnimalParams(
       name="eagle",
       is_planar=False,
       max_speed=30.0,
       max_turn=8.0,
       max_accel=8.0,
       turn_noise=0.4,
       epsilon=0.03,
   )

def pigeon_params():
    return AnimalParams(
        name="pigeon",
        is_planar = False,
        max_speed = 15.0,
        max_turn  = 16.0,
        max_accel = 6.0,
        turn_noise = 0.6,
        epsilon = 0.8,
    )

# Drones
def drone_params():
    return DroneParams(
        name="drone",
        is_planar=False,
        max_speed=12.0,
        max_turn=4.0,
        max_view_yaw=2.0,
        max_accel=4.0,
        camera_pitch=-30
    )
