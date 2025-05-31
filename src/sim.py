import numpy as np
from matplotlib import pyplot as plt
import ground_truth_model
import estimation_model
import plotly.graph_objects as go
import scene
import constants as consts
from ekf import EKF
import imm
import sensor_model
import postpro
import estimation_helpers as eh

'''
TODOs: 
1. Update documentation
2. implement particle filter
'''
# GROUND TRUTH MODEL SETUP
x0 = np.array([0, 0, 1.0, 20.0, 0.0, 5.0])
run_time = 10
dt = 0.01
model = ground_truth_model.SystemModel(x0, run_time, dt)

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
P_ij = np.array([[0.9, 0.1],
                 [0.2, 0.8]])
ekf = EKF(estimation_model.FlightModel(), mu_initial, sigma_initial, Q, R, dt)
ekf_flight = EKF(estimation_model.FlightModel(), mu_initial, sigma_initial, Q, R, dt)
ekf_bounce = EKF(estimation_model.BounceModel(), mu_initial, sigma_initial, Q, R, dt)
imm_tracker = imm.IMMTracker([ekf_flight, ekf_bounce], P_ij, dt)

# RUN SIM
show_cameras = True
t, x = model.run_sim()
y, visibility = sensor_model.get_camera_measurements(cameras, x, 1)
x_est, sigma = eh.run_estimator(ekf, cameras, y, visibility)
x_est_imm, sigma_imm = eh.run_estimator(imm_tracker, cameras, y, visibility)

# PLOT SCENE RESULTS
fig = go.Figure()
scene.draw_court(fig)
scene.plot_ball_trajectory(fig, x, "Ground Truth", "green")
scene.plot_ball_trajectory(fig, x_est, "EKF Estimate", "yellowgreen")
scene.plot_ball_trajectory(fig, x_est_imm, "IMM Estimate", "blue")
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
if model.x_impact is not None:
    x, y = model.x_impact[0], model.x_impact[1]
    scene.plot_impact_location(ax, x, y, None, color='green',
                               label='Ground Truth',
                               show_plot=False)
if ekf.impact_data is not None:
    x, y, sigma = ekf.impact_data
    scene.plot_impact_location(ax, x, y, sigma, color='red',
                               label='EKF',
                               show_plot=False)

if imm_tracker.impact_data is not None:
    x, y, sigma = imm_tracker.impact_data
    scene.plot_impact_location(ax, x, y, sigma, color='blue',
                               label='IMM',
                               show_plot=False)
fig.show()

# PLOT IMM RESULTS
scene.plot_imm_results(x_est_imm, np.asarray(imm_tracker.alpha_hist))

# COLLECT BOUNCE STATISTICS
mu_initial = np.zeros(6)
sigma_initial = np.eye(6)
Q = 0.1 * np.eye(6)
R = 1.0 * np.eye(2 * len(cameras))
ekf.reset(mu_initial, sigma_initial)
mean_error, std_dev, missed_detections = postpro.run_study(num_runs=100,
                                                           ground_truth_model=model,
                                                           estimator=ekf,
                                                           mu_initial=mu_initial,
                                                           sigma_initial=sigma_initial,
                                                           cameras=cameras,
                                                           camera_noise=1.0)
imm_tracker.reset(mu_initial, sigma_initial)
imm_tracker = imm.IMMTracker([ekf_flight, ekf_bounce], P_ij, dt)
postpro.run_study(num_runs=100,
                  ground_truth_model=model,
                  estimator=imm_tracker,
                  mu_initial=mu_initial,
                  sigma_initial=sigma_initial,
                  cameras=cameras,
                  camera_noise=1.0)
