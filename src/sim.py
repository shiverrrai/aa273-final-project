import numpy as np
import system_model
import plotly.graph_objects as go
import scene

x0 = np.array([0, 0, 1.0, 20.0, 0.0, 5.0])
sim_time = 10
model = system_model.SystemModel(x0, sim_time, 0.1)
t, x = model.run_sim()

t = np.asarray(t)
x = np.asarray(x)

# visualize results
fig = go.Figure()
scene.draw_court(fig)
scene.plot_ball_trajectory(fig, x)
scene.show_scene(fig)
fig.show()