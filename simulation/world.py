from simulation.settings import *
from simulation.agents.animals import Eagle, Jackal, Pigeon

class World:
    def __init__(self, seed=None, config=None):
        if seed is not None:
            np.random.seed(seed)

        self.config = config
        self.agents = []

        self.log = []
        self.t = 0.0

    @classmethod
    def random_world(cls, seed=None):
        world = cls(seed)
        world.spawn_random()
        return world

    def random_position(self):
        return np.array([
            uniform(0, MAP_WIDTH),
            uniform(0, MAP_HEIGHT),
            0.0
        ])

    def spawn_random(self):
        for _ in range(EAGLE_COUNT):
            self.agents.append(Eagle(self.random_position()))

        for _ in range(JACKAL_COUNT):
            self.agents.append(Jackal(self.random_position()))

        for _ in range(PIGEON_COUNT):
            self.agents.append(Pigeon(self.random_position()))

    def get_observation(self, agent):
        return {
            "pos": agent.pos.copy(),
            "speed": agent.speed,
            "direction": agent.direction,
            "rng": np.random.normal()
        }

    def step(self, dt):
        for agent in self.agents:
            obs = self.get_observation(agent)

            angular_velocity, accel = agent.update(dt, obs)

            self.log_agent_state(agent, angular_velocity, accel)

        self.t += dt

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.t = 0
        self.done = False

    def log_agent_state(self, agent, angular_velocity, accel):
        vx, vy, vz = agent.direction * agent.speed

        self.log.append({
            "t": self.t,
            "agent_id": agent.agent_id,
            "species": type(agent).__name__,
            "mode": agent.mode,

            # position (3D)
            "x": agent.pos[0],
            "y": agent.pos[1],
            "z": agent.pos[2],

            # velocity
            "vx": vx,
            "vy": vy,
            "vz": vz,

            # control signals (disturbance)
            "speed": agent.speed,
            "angular_velocity": angular_velocity,
            "accel": accel,
        })

    def save_log_csv(self, path):
        if not self.log:
            print("Warning: log is empty, nothing to save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)

        print(f"Saved log to {path}")
