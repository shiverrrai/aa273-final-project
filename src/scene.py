import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse
import constants as consts


# plotting utilities for visualization purposes

def draw_court(fig):
    # Generate grid for tennis court surface
    x_court = np.linspace(0, consts.court_length, 2)
    y_court = np.linspace(-consts.court_width / 2, consts.court_width / 2, 2)
    x_mesh, y_mesh = np.meshgrid(x_court, y_court)
    z_mesh = np.full_like(x_mesh, 0)

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

# Utility function to smooth trajectory using moving average for ball trajectory
def smooth_trajectory(trajectory, window_size=5):
    """Apply moving average smoothing to trajectory."""
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)
        smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
    
    return smoothed

def plot_ball_trajectory(fig, trajectory, name, color, line_style=None):
    # Plot tennis ball trajectory
    if line_style is not None:
        # Apply smoothing
        smooth_traj = smooth_trajectory(trajectory, window_size=5)

        fig.add_trace(go.Scatter3d(
            x=smooth_traj[:, 0], 
            y=smooth_traj[:, 1], 
            z=smooth_traj[:, 2],
            mode='lines',  # Clean lines only, no markers
            name=name,
            line=dict(
                width=5,      # Thick lines for visibility
                color=color   # Remove dash - all solid lines
            ),
            showlegend=True,
        ))
    else:
        # Original behavior (lines + markers) - keeps sim.py working
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines+markers',
            name=name,
            marker=dict(size=3, color=color),
            line=dict(width=4),
            showlegend=True,
        ))


def draw_camera_frustum(fig, cam, color, cam_name, near_plane=5, far_plane=25):
    pos = cam.C.flatten()
    R_inv = cam.R.T # inverse: camera --> world

    # camera center (no legend)
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers',
        marker=dict(size=12, color=color, symbol='diamond'),
        showlegend=False
    ))

    # frustum plane at near distance only
    w = near_plane * cam.image_size[0] / cam.f
    h = near_plane * cam.image_size[1] / cam.f
    corners_cam = np.array([
        [-w / 2, -h / 2, -near_plane],
        [w / 2, -h / 2, -near_plane],
        [w / 2, h / 2, -near_plane],
        [-w / 2, h / 2, -near_plane],
    ])
    near_plane_world = [
        (R_inv @ c.reshape(3, 1) + cam.C).flatten() for c in corners_cam
    ]
    near_plane_world = np.array(near_plane_world)

    # center→near lines (the frustum edges)
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[pos[0], near_plane_world[i, 0]],
            y=[pos[1], near_plane_world[i, 1]],
            z=[pos[2], near_plane_world[i, 2]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

    # outline of near plane
    loop = np.vstack([near_plane_world, near_plane_world[0]])
    fig.add_trace(go.Scatter3d(
        x=loop[:, 0],
        y=loop[:, 1],
        z=loop[:, 2],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))

    # optical axis → “view direction”
    mid = (near_plane + far_plane) / 2
    end = (R_inv @ np.array([0, 0, -mid]).reshape(3, 1) + cam.C).flatten()
    fig.add_trace(go.Scatter3d(
        x=[pos[0], end[0]],
        y=[pos[1], end[1]],
        z=[pos[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=f"{cam_name} view direction",
        showlegend=True
    ))


def generate_camera_views(cameras, positions, model_impacts=None):
    """
    Generate and display camera views showing the projected ball trajectory.
    Note: The camera view depends on the camera's roll, pitch, and yaw angles 
    defined in the PinholeCamera object.
    
    Args:
        cameras (list): List of PinholeCamera objects to generate views for
        positions (ndarray): Nx3 array of ball positions in world coordinates
        model_impacts (list, optional): List of impact states for bounce
        visualization
    
    Returns:
        None: Displays matplotlib figures for each camera view
    """
    print("\nGenerating camera views...")

    for i, cam in enumerate(cameras):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up image coordinates
        ax.set_xlim(0, cam.image_size[0])
        ax.set_ylim(cam.image_size[1],
                    0)  # Inverted y-axis for image coordinates
        ax.set_facecolor('darkgreen')

        # Add court reference lines
        ax.axhline(y=cam.image_size[1] // 2, color='white', linewidth=2,
                   alpha=0.4)
        ax.axvline(x=cam.image_size[0] // 2, color='white', linewidth=2,
                   alpha=0.4)

        # Plot trajectory points that are visible
        visible_pixels = []
        visible_indices = []

        for j, point in enumerate(positions):
            world_coords = point[:3] if len(point) > 3 else point
            pixel, visible = cam.g(world_coords)
            if visible:
                visible_pixels.append(pixel)
                visible_indices.append(j)

        if visible_pixels:
            visible_pixels = np.array(visible_pixels)

            # Plot trajectory path
            ax.plot(visible_pixels[:, 0], visible_pixels[:, 1],
                    'yellow', linewidth=3, alpha=0.7, label='Ball trajectory')

            # Mark key points: start and end of visible trajectory
            ax.scatter(visible_pixels[0, 0], visible_pixels[0, 1],
                       c='lime', s=150, marker='^', edgecolors='black',
                       linewidth=2,
                       label='Start', zorder=10)

            if len(visible_pixels) > 1:
                ax.scatter(visible_pixels[-1, 0], visible_pixels[-1, 1],
                           c='red', s=150, marker='v', edgecolors='black',
                           linewidth=2,
                           label='End', zorder=10)

        # Mark bounce points if visible
        if model_impacts is not None:
            for bounce_idx, bounce_point in enumerate(model_impacts):
                bounce_pixel, bounce_visible = cam.g(bounce_point[:3])
                if bounce_visible:
                    ax.scatter(bounce_pixel[0], bounce_pixel[1],
                               c='orange', s=200, marker='X',
                               edgecolors='black', linewidth=2,
                               label=f'Bounce {bounce_idx + 1}', zorder=10)

        # Calculate coverage statistics
        coverage = 100 * len(visible_pixels) / len(positions) if len(
            positions) > 0 else 0

        # Add title and labels
        ax.set_title(f'Camera {i + 1} View - {coverage:.1f}% Coverage',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Pixel U', fontsize=12)
        ax.set_ylabel('Pixel V', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add legend and coverage info
        ax.legend(loc='upper right', framealpha=0.9)
        coverage_text = (f'Visible: {len(visible_pixels)}/{len(positions)} '
                         f'points')
        ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)

        plt.tight_layout()
        plt.show()

    print("Camera views generated!")


def show_scene(fig):
    fig.update_layout(
        scene=dict(
            xaxis_title='X (court length, m)',
            yaxis_title='Y (court width, m)',
            zaxis_title='Z (height, m)',
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5),
        ),
        title='Tennis Ball Trajectory Results',
        margin=dict(l=0, r=0, b=0, t=30),
    )


def compute_confidence_ellipse(x, y, sigma, color='red', label=''):
    if sigma is None:
        return None

    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Compute angle of the ellipse in degrees
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 95% confidence interval -> scale eigenvalues by sqrt(5.991)
    width, height = 2 * np.sqrt(
        vals * 5.991)  # chi^2 with 2 DOF, 95% CI

    # Draw the confidence ellipse
    ellipse = Ellipse((x, y), width, height, angle=angle,
                      edgecolor=color,
                      facecolor='none', linestyle='--', linewidth=2,
                      label=label + ' Confidence Ellipse')
    return ellipse


def plot_impact_location(ax, x, y, sigma, color='green',
                         label='', show_plot=True):
    # Ensure aspect ratio is 1:1 so circles/ellipses look correct
    ax.set_aspect('equal')

    # Plot ball location
    circle = Circle((x, y), consts.r, color=color, alpha=0.6,
                    edgecolor='black', label=label + ' Bounce Location')
    ax.add_patch(circle)

    confidence_ellipse = compute_confidence_ellipse(x, y, sigma, color, label)
    w, h = 0, 0
    if confidence_ellipse is not None:
        ax.add_patch(confidence_ellipse)
        w, h = confidence_ellipse.width, confidence_ellipse.height

    # Set plot limits with padding
    max_dim = max(consts.r, w / 2, h / 2)
    padding = 0.5 + max_dim  # add padding around ball and ellipse

    ax.set_xlim(x - padding, x + padding)
    ax.set_ylim(y - padding, y + padding)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Tennis Ball Bounce Location")
    ax.legend()
    ax.grid(True)

    if show_plot:
        plt.show()


def plot_imm_results(state, alpha):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    index = np.arange(state.shape[0])
    ax1.plot(index, state[:, 2], label='ball vertical position', color='red')
    ax2.plot(index, alpha[:, 0], label='flight probability', color='blue')
    ax2.plot(index, alpha[:, 1], label='bounce probability', color='green')
    ax1.legend()
    ax2.legend()
    ax1.set_title("IMM Belief vs Ball Vertical Position")
    ax1.set_xlabel("index")
    ax1.set_ylabel("Ball Vertical Position (m)")
    ax2.set_ylabel("IMM Belief")
    ax1.grid(True)
    plt.show()
