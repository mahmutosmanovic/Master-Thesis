import csv
import math
from collections import defaultdict

import pyray as pr  # type: ignore


CSV_PATH = "logs/simulations/test_jackal_random_single.csv"
MODEL_PATH = "view/assets/source/model.glb"
TEXTURE_PATH = "view/assets/textures/gltf_embedded_0.jpeg"
FLOOR_TEXTURE_PATH = "view/assets/textures/green.png"

SCREEN_W = 1280
SCREEN_H = 720
FPS = 60


def load_csv(path):
    frames = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t"])
            frames[t].append(row)
    times = sorted(frames.keys())
    return frames, times


def update_camera_controls(dt, cam_distance, cam_yaw, cam_pitch, cam_roll):
    rot_speed = 1.5 * dt
    zoom_speed = 5 * dt

    if pr.is_key_down(pr.KEY_A): cam_yaw -= rot_speed
    if pr.is_key_down(pr.KEY_D): cam_yaw += rot_speed
    if pr.is_key_down(pr.KEY_W): cam_pitch += rot_speed
    if pr.is_key_down(pr.KEY_S): cam_pitch -= rot_speed

    if pr.is_key_down(pr.KEY_Q): cam_roll -= rot_speed
    if pr.is_key_down(pr.KEY_E): cam_roll += rot_speed

    if pr.is_key_down(pr.KEY_UP): cam_distance -= zoom_speed
    if pr.is_key_down(pr.KEY_DOWN): cam_distance += zoom_speed

    cam_distance = max(2.0, min(30.0, cam_distance))
    cam_pitch = max(-1.5, min(1.5, cam_pitch))

    return cam_distance, cam_yaw, cam_pitch, cam_roll


def apply_camera_transform(camera, cam_target, cam_distance, cam_yaw, cam_pitch, cam_roll):
    camera.position.x = cam_target.x + cam_distance * math.cos(cam_pitch) * math.sin(cam_yaw)
    camera.position.y = cam_target.y + cam_distance * math.sin(cam_pitch)
    camera.position.z = cam_target.z + cam_distance * math.cos(cam_pitch) * math.cos(cam_yaw)

    camera.target = cam_target
    camera.up.x = math.sin(cam_roll)
    camera.up.y = math.cos(cam_roll)
    camera.up.z = 0.0


class Floor:
    def __init__(self, texture_path):
        self.texture = pr.load_texture(texture_path)
        mesh = pr.gen_mesh_plane(640, 640, 1, 1)
        self.model = pr.load_model_from_mesh(mesh)
        pr.set_material_texture(self.model.materials[0], pr.MATERIAL_MAP_ALBEDO, self.texture)
        self.position = pr.Vector3(0, -1, 0)

    def draw(self):
        pr.draw_model(self.model, self.position, 1.0, pr.WHITE)

    def unload(self):
        pr.unload_texture(self.texture)
        pr.unload_model(self.model)


class Jackal:
    def __init__(self, model_path, texture_path):
        self.model = pr.load_model(model_path)
        self.texture = pr.load_texture(texture_path)

        pr.set_material_texture(
            self.model.materials[0],
            pr.MATERIAL_MAP_ALBEDO,
            self.texture,
        )

        self.position = pr.Vector3(0, 0, 0)
        self.scale = 1.0

    def move(self, x, y, z):
        self.position.x = x
        self.position.y = y
        self.position.z = z

    def draw(self):
        pr.draw_model(self.model, self.position, self.scale, pr.WHITE)

    def unload(self):
        pr.unload_texture(self.texture)
        pr.unload_model(self.model)


pr.set_target_fps(FPS)
pr.init_window(SCREEN_W, SCREEN_H, "Stealth-Fleet")

frames, times = load_csv(CSV_PATH)

camera = pr.Camera3D()
camera.target = pr.Vector3(0, 0, 0)
camera.fovy = 45.0
camera.projection = pr.CAMERA_PERSPECTIVE

cam_target = pr.Vector3(0, 0, 0)
cam_distance = 8.0
cam_yaw = 0.0
cam_pitch = 0.5
cam_roll = 0.0

floor = Floor(FLOOR_TEXTURE_PATH)
jackal = Jackal(MODEL_PATH, TEXTURE_PATH)

while not pr.window_should_close():
    dt = pr.get_frame_time()

    cam_distance, cam_yaw, cam_pitch, cam_roll = update_camera_controls(
        dt, cam_distance, cam_yaw, cam_pitch, cam_roll
    )

    apply_camera_transform(
        camera, cam_target, cam_distance, cam_yaw, cam_pitch, cam_roll
    )

    pr.begin_drawing()
    pr.clear_background(pr.WHITE)

    pr.begin_mode_3d(camera)

    floor.draw()
    jackal.draw()

    pr.end_mode_3d()

    pr.draw_text("WASD rotate | Q/E roll | Arrows zoom", 10, 10, 20, pr.DARKGRAY)

    pr.end_drawing()

    # loop

floor.unload()
jackal.unload()
pr.close_window()
