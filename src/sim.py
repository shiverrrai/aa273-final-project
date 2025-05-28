import numpy as np
import system_model
import plotly.graph_objects as go
import scene
import constants as consts
from ekf import EKF
import sensor_model


# GROUND TRUTH MODEL SETUP
x0 = np.array([0, 0, 1.0, 20.0, 0.0, 5.0])
run_time = 10
dt = 0.01
model = system_model.SystemModel(x0, run_time, dt)

# CAMERA MODEL SETUP
camera_params = {
    'distance': 25,  # Meters from court centerline
    'elevation': 4,  # Meters above ground
    'focal_length': 1000,  # Pixels (higher = more zoom, narrower FOV)
    'image_size': (1280, 960),  # image size in pixels
}
camera1 = sensor_model.PinholeCamera(
    position=[consts.court_length / 2, -camera_params['distance'],
              camera_params['elevation']],
    rotation_angles=[-np.pi / 2, 0, 0],  # Roll (x), pitch (y), yaw (z)
    focal_length=camera_params['focal_length'],
    image_size=camera_params['image_size'],
)
camera2 = sensor_model.PinholeCamera(
    position=[consts.court_length / 2, camera_params['distance'],
              camera_params['elevation']],
    rotation_angles=[np.pi / 2, 0, 0],  # Roll (x), pitch (y), yaw (z)
    focal_length=camera_params['focal_length'],
    image_size=camera_params['image_size'],
)
cameras = [camera1, camera2]

# ESTIMATION ALGORITHM SETUP
mu = np.zeros(6)
sigma = np.eye(6)
Q = 0.1 * np.eye(6)
R = np.eye(2 * len(cameras))
ekf = EKF(mu, sigma, Q, R, dt)

# RUN SIM
t, x = model.run_sim()
y, _ = sensor_model.get_camera_measurements(cameras, x)
x_est, sigma = ekf.run(cameras, y)

# PLOT RESULTS
fig = go.Figure()
scene.draw_court(fig)
scene.plot_ball_trajectory(fig, x, "Ground Truth", "green")
scene.plot_ball_trajectory(fig, x_est, "EKF Estimate", "yellowgreen")
colors = ['red', 'blue']
for i, cam in enumerate(cameras):
    scene.draw_camera_frustum(
        fig, cam, colors[i], f"Camera {i+1}",
        near_plane=5, far_plane=20
    )
scene.show_scene(fig)

fig.show(renderer="browser")
