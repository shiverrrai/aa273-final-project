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

# visualize trajectory
fig = go.Figure()
scene.draw_court(fig)
scene.plot_ball_trajectory(fig, x)
scene.show_scene(fig)

# Show the plot as popup window and suppress verbose output
fig.show(renderer="browser");

# display ball bounce location
assert len(model.x_impact) > 0, "ball did not bounce!"
x_impact = np.asarray(model.x_impact)
xf, yf, zf = x_impact[0, 0:3]
scene.plot_impact_location((xf, yf), system_model.r)