from environment.config import EnvConfig, AnimalParams, DroneParams
from environment.agents.behaviour import BehaviourConfig, CRWConfig, ExploreExploitConfig
from environment.agents.sensor import SensorConfig, CameraConfig, GPSConfig

def rand5jack1drone():
    return EnvConfig(
        # simulation
        dt=0.2,
        max_t=100.0,

        # map spawning bounds
        map_width=200.0,
        map_height=200.0,
        map_altitude=100.0,

        # animals
        animals=[
            dict(params=jackal_params(), count=1, behaviour_cfg=ExploreExploitConfig()),
            # dict(params=jackal_params(), count=1, behaviour_cfg=CRWConfig()),
            # dict(params=eagle_params(),  count=0, behaviour_cfg=CRWConfig()),
            # dict(params=pigeon_params(), count=0, behaviour_cfg=CRWConfig()),
        ],

        resource_frequency = 0.006,
        resource_scale = 0.3,
        resource_abundance = 0.4,

        # drones
        drones=[
            dict(params=drone_params(), count=1, sensor_cfg=CameraConfig(far_plane=200.0)),
        ],

        # observation and reward
        distance_scale = 3,
        alignment_scale = 1,
        disturbance_scale = 2.5,
        control_scale = 0.2
    )

# Animals (placeholders, should be changed to 3 general "animals")
def jackal_params():
    return AnimalParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        avoidance_threshold=0.75,
        flight_threshold=1,
    )

def eagle_params():
   return AnimalParams(
        name="eagle",
        is_planar=False,
        max_speed=30.0,
        max_turn=8.0,
        avoidance_threshold=0.75,
        flight_threshold=1,
   )

def pigeon_params():
    return AnimalParams(
        name="pigeon",
        is_planar = False,
        max_speed = 15.0,
        max_turn  = 16.0,
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
        max_view_yaw=2,
        camera_pitch=-30
    )
