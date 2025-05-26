import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


# plotting utilities for visualization purposes

def draw_court(fig):
    # Tennis court surface
    court_length = 23.77 # m
    court_width = 8.23 # m
    court_height = 0  # flat ground

    # Generate grid for surface
    x_court = np.linspace(0, court_length, 2)
    y_court = np.linspace(-court_width / 2, court_width / 2, 2)
    x_mesh, y_mesh = np.meshgrid(x_court, y_court)
    z_mesh = np.full_like(x_mesh, court_height)

    # Add court surface
    fig.add_trace(go.Surface(
        x=x_mesh,
        y=y_mesh,
        z=z_mesh,
        colorscale=[[0, 'blue'], [1, 'blue']],
        opacity=0.5,
        showscale=False,
        name='Court'
    ))
    return fig

def plot_ball_trajectory(fig, trajectory):
    # Plot tennis ball trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
        mode='lines+markers',
        name='Ball',
        marker=dict(size=3, color='yellowgreen'),
        line=dict(width=4)
    ))

def show_scene(fig):
    fig.update_layout(
        scene=dict(
            xaxis_title='X (court length, m)',
            yaxis_title='Y (court width, m)',
            zaxis_title='Z (height, m)',
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5)
        ),
        title='Ground Truth Tennis Ball Trajectory',
        margin=dict(l=0, r=0, b=0, t=30)
    )

def plot_impact_location(loc, r):
    x, y = loc
    fig, ax = plt.subplots()
    ax.set_aspect('equal')  # Ensure aspect ratio is 1:1 so circles look correct
    circle = Circle((x, y), r, color='green', alpha=0.6,
                    edgecolor='black')
    ax.add_patch(circle)
    padding = 0.1  # for visualization padding around the ball
    ax.set_xlim(x - r - padding, x + r + padding)
    ax.set_ylim(y - r - padding, y + r + padding)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Tennis Ball Impact Point")
    plt.grid(True)
    plt.show()

def draw_camera_frustum(fig, cam, color, cam_name, near_plane=5, far_plane=25):
    pos = cam.C.flatten()
    R_inv = cam.R.T

    # camera center (no legend)
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers',
        marker=dict(size=12, color=color, symbol='diamond'),
        showlegend=False
    ))

    # frustum planes at near and far (no legend)
    planes = []
    for depth in (near_plane, far_plane):
        w = depth * cam.image_size[0] / cam.f
        h = depth * cam.image_size[1] / cam.f
        corners_cam = np.array([
            [-w/2, -h/2, -depth],
            [ w/2, -h/2, -depth],
            [ w/2,  h/2, -depth],
            [-w/2,  h/2, -depth],
        ])
        plane_world = [(R_inv @ c.reshape(3,1) + cam.C).flatten() for c in corners_cam]
        planes.append(np.array(plane_world))

    # edges
    for i in range(4):
        # near→far
        fig.add_trace(go.Scatter3d(
            x=[planes[0][i,0], planes[1][i,0]],
            y=[planes[0][i,1], planes[1][i,1]],
            z=[planes[0][i,2], planes[1][i,2]],
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
            showlegend=False
        ))
        # center→near
        fig.add_trace(go.Scatter3d(
            x=[pos[0], planes[0][i,0]],
            y=[pos[1], planes[0][i,1]],
            z=[pos[2], planes[0][i,2]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

    # outlines of near & far
    for plane in planes:
        loop = np.vstack([plane, plane[0]])
        fig.add_trace(go.Scatter3d(
            x=loop[:,0],
            y=loop[:,1],
            z=loop[:,2],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

    # optical axis → “view direction”
    mid = (near_plane + far_plane) / 2
    end = (R_inv @ np.array([0, 0, -mid]).reshape(3,1) + cam.C).flatten()
    fig.add_trace(go.Scatter3d(
        x=[pos[0], end[0]],
        y=[pos[1], end[1]],
        z=[pos[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=f"{cam_name} view direction",
        showlegend=True
    ))


