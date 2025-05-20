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


