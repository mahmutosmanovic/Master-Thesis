from environment import EnvConfig, AnimalConfig, DroneConfig, CRWConfig, ExploreExploitConfig, TraplineConfig

def standard_env():
    return EnvConfig(
        # simulation
        dt=0.2,      # (seconds)
        max_t=100.0, # (seconds)

        # map spawning bounds
        map_size=1000.0,    # (meters)
        map_altitude=100.0, # (meters)

        # Animal settings
        force_bounds = False, # restrict animal movement to map bounds ([0, map_size] and [0, map_size], reflection on edges)
        # each entry: {params: AnimalParams, count: int, behaviour: BehaviourConfig}
        animals=[
            dict(config=standard_animal(),              # AnimalConfig
                 count=1,                               # Number of animals of this type
                 behaviour_cfg=ExploreExploitConfig()), # BehaviourConfig
        ],

        # Resource map parameters (all animals share resource map)
        p_wavelenght = 200.0, # (meters) wavelenght of major resource noise
        p_reduction = 0.2,    # Reduction on raw encounter probability
        p_scale = 0.4,        # Scaling of reduced probability
        sample_res = 5.0,     # (meters per sample) Sample resolution for poi generation
        min_poi_p = 1e-2,     # minimum value for a local maxima to be considered a poi
        kernel_size = 250.0,  # (meters) kernel size for poi generation (local maxima)

        # drones
        drone_target_order = "random", # Drone target order ("round_robin", "random"), selects how to assign drones to animals
        # each entry: {params: DroneParams, count: int, spawn_range: [float, float]}
        drones=[
            dict(config=standard_drone(),     # DroneConfig
                 count=1,                     # Number of animals of this type
                 spawn_range=[100.0, 150.0]), # [nearest spawnable distance, furthest spawnable distance]     
        ],

        # observation and reward
        distance_scale = 3,
        alignment_scale = 1,
        disturbance_scale = 2.5,
        control_scale = 0.2,
    )

# Animals (placeholders, should be changed to 3 general "animals")
def standard_animal():
    return AnimalConfig(
        name="standard_animal",   # animal type name
        is_planar=True,           # restricted to z=0?
        max_speed=12.0,           # (m/s) maximum speed
        avoidance_threshold=0.75, # disturbance threshold to initiate avoiding behaviour
        flight_threshold=1,       # disturbance threshold to initiate fleeing behaviour
    )

# Drones
def standard_drone():
    return DroneConfig(
        name="standard_drone", # drone type name
        is_planar=False,       # restricted to z=0?
        max_speed=12.0,        # (m/s) maximum speed
        max_view_yaw=2,        # (radians) maximum view rotation speed
        camera_pitch=-30,      # (degrees) camera pitch (0 -> horizontal, -90 -> straight down, 90 -> straight up)
        hfov = 90,             # (degrees) horizontal field of view
        vfov = 56,             # (degrees) vertical field of view
        near_plane = 1,        # (meters) frustum near plane distance
        far_plane = 200,       # (meters) frustum far plane distance
        max_targets = 1,       # Number of slots in observation, if lower than number of animals includes nearest max_targets
    )
