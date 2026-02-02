from settings import *
from utils import *
from view import *
from pigeon import *
from logger import *
from drone import *
from agent import *

def main(train=True):
    # df = pd.read_csv(DATA_FOLDER_PATH)
    # df = transform_to_epsg32636(df)

    pigeon_config = PigeonConfig()
    drone_config = DroneConfig()

    pigeon = Pigeon(pigeon_config, BEHAVIOR, start_pos=(0,0,0))
    drone = Drone(drone_config, (pigeon.x, pigeon.y, pigeon.z))
    
    obs_dim = 3 # obs = (in_view, angle, dist) -> 3
    act_dim = 4 # action = dx,dy,dz,dyaw -> 4
    agent = PPOAgent(obs_dim, act_dim)

    logger = Logger()
    for t in range(STEPS):
        pigeon.step()
        logger.write(CSV_PATH, t, "PIGEON", (pigeon.x,pigeon.y,pigeon.z,0))

        observation = drone.observe(pigeon)
        action = drone.policy(observation)
        drone.step(action)
        logger.write(CSV_PATH, t, "DRONE", (drone.x,drone.y,drone.z,drone.yaw))


    df_dynamic = pd.read_csv(CSV_PATH)
    draw_trail_3D(df_dynamic, interval=100)

if __name__ == "__main__":
    main()
