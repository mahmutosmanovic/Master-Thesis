import csv
import math
from collections import defaultdict, deque

import pyray as pr  # type: ignore

# --------------------------------------------------
# PATHS & CONSTANTS
# --------------------------------------------------
CSV_PATH = "logs/simulations/test_jackal_random_single.csv"

PIGEON_MODEL_PATH = "view/assets/source/pigeon.glb"
EAGLE_MODEL_PATH = "view/assets/source/eagle.glb"
JACKAL_MODEL_PATH = "view/assets/source/jackal.glb"
TEXTURE_PATH = "view/assets/textures/gltf_embedded_0.jpeg"

FLOOR_TEXTURE_PATH = "view/assets/textures/light.png"

SCREEN_W = 1280
SCREEN_H = 720
FPS = 60

MAX_TRAIL_POINTS = 35
TRAIL_SIZE = 0.1
TRAIL_Y_OFFSET = -0.6

# --------------------------------------------------
# CSV LOADING
# --------------------------------------------------
def load_csv(path: str):
    frames = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t"])
            frames[t].append(row)

    times = sorted(frames.keys())
    return frames, times


def get_position_from_row(row):
    # Simulation uses X Z Y → convert to Raylib X Y Z
    return (
        float(row["x"]),
        float(row["z"]),
        float(row["y"]),
    )


def get_velocity_from_row(row):
    return (
        float(row["vx"]),
        float(row["vz"]),
    )


# --------------------------------------------------
# CAMERA CONTROLS
# --------------------------------------------------
def update_camera_controls(
    dt,
    cam_distance,
    cam_yaw,
    cam_pitch,
    cam_roll,
):
    rot_speed = 1.5 * dt
    zoom_speed = 5.0 * dt

    if pr.is_key_down(pr.KEY_A):
        cam_yaw -= rot_speed
    if pr.is_key_down(pr.KEY_D):
        cam_yaw += rot_speed
    if pr.is_key_down(pr.KEY_W):
        cam_pitch += rot_speed
    if pr.is_key_down(pr.KEY_S):
        cam_pitch -= rot_speed
    if pr.is_key_down(pr.KEY_Q):
        cam_roll -= rot_speed
    if pr.is_key_down(pr.KEY_E):
        cam_roll += rot_speed
    if pr.is_key_down(pr.KEY_UP):
        cam_distance -= zoom_speed
    if pr.is_key_down(pr.KEY_DOWN):
        cam_distance += zoom_speed

    cam_distance = max(3.0, min(30.0, cam_distance))
    cam_pitch = max(-1.2, min(1.2, cam_pitch))

    return cam_distance, cam_yaw, cam_pitch, cam_roll


def apply_camera_transform(
    camera,
    cam_target,
    cam_distance,
    cam_yaw,
    cam_pitch,
    cam_roll,
):
    camera.position.x = (
        cam_target.x
        + cam_distance * math.cos(cam_pitch) * math.sin(cam_yaw)
    )
    camera.position.y = (
        cam_target.y
        + cam_distance * math.sin(cam_pitch)
    )
    camera.position.z = (
        cam_target.z
        + cam_distance * math.cos(cam_pitch) * math.cos(cam_yaw)
    )

    camera.target = cam_target
    camera.up = pr.Vector3(
        math.sin(cam_roll),
        math.cos(cam_roll),
        0.0,
    )


# --------------------------------------------------
# FLOOR
# --------------------------------------------------
class Floor:
    def __init__(self, texture_path, center_x, center_z):
        self.texture = pr.load_texture(texture_path)
        pr.set_texture_wrap(self.texture, pr.TEXTURE_WRAP_REPEAT)

        mesh = pr.gen_mesh_plane(2000, 2000, 1, 1)
        self.model = pr.load_model_from_mesh(mesh)

        pr.set_material_texture(
            self.model.materials[0],
            pr.MATERIAL_MAP_ALBEDO,
            self.texture,
        )

        self.position = pr.Vector3(center_x, -1.0, center_z)

    def draw(self):
        pr.draw_model(self.model, self.position, 1.0, pr.WHITE)

    def unload(self):
        pr.unload_texture(self.texture)
        pr.unload_model(self.model)


# --------------------------------------------------
# JACKAL
# --------------------------------------------------
class Jackal:
    def __init__(self, model_path, texture_path, initial_position):
        self.model = pr.load_model(model_path)
        self.texture = pr.load_texture(texture_path)

        pr.set_material_texture(
            self.model.materials[0],
            pr.MATERIAL_MAP_ALBEDO,
            self.texture,
        )

        self.position = initial_position
        self.scale = 1.0
        self.yaw = 0.0

        self.trail = deque(maxlen=MAX_TRAIL_POINTS)

    def move(self, x, y, z, vx, vz):
        # Store trail
        self.trail.appendleft(
            pr.Vector3(
                self.position.x,
                self.position.y,
                self.position.z,
            )
        )

        # Update position
        self.position.x = x
        self.position.y = y
        self.position.z = z

        # Update facing direction
        if abs(vx) > 1e-4 or abs(vz) > 1e-4:
            self.yaw = math.atan2(vx, vz)

    def draw(self):
        # Draw movement trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (1.0 - i / MAX_TRAIL_POINTS))
            color = pr.Color(152,119,76, alpha)

            trail_pos = pr.Vector3(
                pos.x,
                pos.y + TRAIL_Y_OFFSET,
                pos.z,
            )

            pr.draw_sphere(trail_pos, TRAIL_SIZE, color)

        # Draw jackal
        axis = pr.Vector3(0.0, 1.0, 0.0)
        angle_deg = math.degrees(self.yaw) + 15

        pr.draw_model_ex(
            self.model,
            self.position,
            axis,
            angle_deg,
            pr.Vector3(self.scale, self.scale, self.scale),
            pr.WHITE,
        )

    def unload(self):
        pr.unload_texture(self.texture)
        pr.unload_model(self.model)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
pr.set_target_fps(FPS)
pr.init_window(SCREEN_W, SCREEN_H, "Stealth-Fleet")

frames, times = load_csv(CSV_PATH)

# Initial state
first_row = frames[times[0]][0]
x0, y0, z0 = get_position_from_row(first_row)

initial_position = pr.Vector3(x0, y0, z0)

camera = pr.Camera3D()
camera.fovy = 45.0
camera.projection = pr.CAMERA_PERSPECTIVE

cam_target = pr.Vector3(x0, y0, z0)
cam_distance = 10.0
cam_yaw = 0.0
cam_pitch = 0.4
cam_roll = 0.0

floor = Floor(FLOOR_TEXTURE_PATH, x0, z0)
eagle = Jackal(EAGLE_MODEL_PATH, TEXTURE_PATH, initial_position)
jackal = Jackal(JACKAL_MODEL_PATH, TEXTURE_PATH, initial_position)
pigeon = Jackal(PIGEON_MODEL_PATH, TEXTURE_PATH, initial_position)

sim_time = times[0]
time_index = 0

while not pr.window_should_close():
    dt = pr.get_frame_time()
    sim_time += dt

    # Advance simulation
    while time_index < len(times) - 1 and sim_time >= times[time_index]:
        row = frames[times[time_index]][0]

        x, y, z = get_position_from_row(row)
        vx, vz = get_velocity_from_row(row)

        jackal.move(x, y, z, vx, vz)

        cam_target.x = x
        cam_target.y = y
        cam_target.z = z

        time_index += 1

    cam_distance, cam_yaw, cam_pitch, cam_roll = update_camera_controls(
        dt,
        cam_distance,
        cam_yaw,
        cam_pitch,
        cam_roll,
    )

    apply_camera_transform(
        camera,
        cam_target,
        cam_distance,
        cam_yaw,
        cam_pitch,
        cam_roll,
    )

    pr.begin_drawing()
    pr.clear_background(pr.WHITE)

    pr.begin_mode_3d(camera)
    floor.draw()
    jackal.draw()
    pr.end_mode_3d()

    pr.draw_text(
        "WASD rotate | Q/E roll | Arrows zoom",
        10,
        10,
        20,
        pr.DARKGRAY,
    )

    pr.end_drawing()

# Cleanup
floor.unload()
jackal.unload()
pr.close_window()
