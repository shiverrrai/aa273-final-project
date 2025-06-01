import numpy as np
from matplotlib import pyplot as plt
import system_model
import plotly.graph_objects as go
import scene
import constants as consts
from ekf import EKF
import sensor_model
import postpro

'''
TODOs: 
2. implement particle filter
3. implement IMM model
'''
# GROUND TRUTH MODEL SETUP
x0 = np.array([0, 0, 1.0, 20.0, 0.0, 5.0])
run_time = 10 # seconds
dt = 0.01 # seconds
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
    rotation_angles=[np.pi / 2, 0, 0],  # Roll (x), pitch (y), yaw (z)
    focal_length=camera_params['focal_length'],
    image_size=camera_params['image_size'],
)
camera2 = sensor_model.PinholeCamera(
    position=[consts.court_length / 2, camera_params['distance'],
              camera_params['elevation']],
    rotation_angles=[-np.pi / 2, 0, 0],  # Roll (x), pitch (y), yaw (z)
    focal_length=camera_params['focal_length'],
    image_size=camera_params['image_size'],
)
cameras = [camera1, camera2]

# ESTIMATION ALGORITHM SETUP
mu_initial = np.zeros(6)
sigma_initial = np.eye(6)
Q = 0.1 * np.eye(6)
R = np.eye(2 * len(cameras))
ekf = EKF(mu_initial, sigma_initial, Q, R, dt)

# RUN SIM
show_cameras = True
t, x = model.run_sim()
y, visibility = sensor_model.get_camera_measurements(cameras, x)
x_est, sigma = ekf.run(cameras, y, visibility)

# PLOT SCENE RESULTS
fig = go.Figure()
scene.draw_court(fig)
scene.plot_ball_trajectory(fig, x, "Ground Truth", "green")
scene.plot_ball_trajectory(fig, x_est, "EKF Estimate", "yellowgreen")
if show_cameras:
    colors = ['red', 'blue']
    for i, cam in enumerate(cameras):
        scene.draw_camera_frustum(
            fig, cam, colors[i], f"Camera {i + 1}",
            near_plane=5, far_plane=20
        )
scene.show_scene(fig)

fig.show(renderer="browser")

# PLOT BOUNCE RESULTS
fig, ax = plt.subplots()
has_impact_data = False
if model.x_impact is not None:
    x, y = model.x_impact[0], model.x_impact[1]
    scene.plot_impact_location(ax, x, y, None, color='green',
                               label='Ground Truth',
                               show_plot=False)
    has_impact_data = True
if ekf.impact_data is not None:
    x, y, sigma = ekf.impact_data
    scene.plot_impact_location(ax, x, y, sigma, color='red',
                               label='EKF',
                               show_plot=False)
    has_impact_data = True
if has_impact_data:
    plt.show()  # Use plt.show() instead of fig.show()
else:
    print("No bounce data detected to plot")

# COLLECT BOUNCE STATISTICS
mean_error, std_dev, missed_detections = postpro.run_study(num_runs=100,
                                                           ground_truth_model=model,
                                                           estimation_filter=ekf,
                                                           mu_initial=mu_initial,
                                                           sigma_initial=sigma_initial,
                                                           cameras=cameras)
