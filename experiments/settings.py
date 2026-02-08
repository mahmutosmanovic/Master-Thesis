from environment.config import EnvConfig, AnimalParams, DroneParams

def rand5jack1drone():
    return EnvConfig(
        # simulation
        dt=0.2,
        max_t=1000.0,

        # map
        map_width=200.0,
        map_height=200.0,
        map_altitude=100.0,

        # POIs
        poi_count=3,
        poi_points=[
            (30.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 30.0, 30.0),
        ],

        # animals
        animals=[
            dict(params=jackal_params(), count=5, mode="random"),
            dict(params=eagle_params(),  count=0, mode="random"),
            dict(params=pigeon_params(), count=0, mode="path_follow"),
        ],

        # drones
        drones=[
            dict(params=drone_params(), count=1, sensor="camera"),
        ],

        # observation and reward
        sensor_scale = 200.0,
        distance_scale = 2.5,
        alignment_scale = 1.0,
        disturbance_scale = 2.5,
    )

# Animals (placeholders, should be changed to 3 general "animals")
def jackal_params():
    return AnimalParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        turn_noise=0.4,
        epsilon=1,
        avoidance_threshold=0.75,
        flight_threshold=1,
    )

def eagle_params():
   return AnimalParams(
        name="eagle",
        is_planar=False,
        max_speed=30.0,
        max_turn=8.0,
        turn_noise=0.4,
        epsilon=0.03,
        avoidance_threshold=0.75,
        flight_threshold=1,
   )

def pigeon_params():
    return AnimalParams(
        name="pigeon",
        is_planar = False,
        max_speed = 15.0,
        max_turn  = 16.0,
        turn_noise = 0.6,
        epsilon = 0.8,
        avoidance_threshold=0.75,
        flight_threshold=1,
    )

# Drones
def drone_params():
    return DroneParams(
        name="drone",
        is_planar=False,
        max_speed=12.0,
        max_turn=4.0,
        max_view_yaw=2.0,
        camera_pitch=-30
    )
