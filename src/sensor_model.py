# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import simulation modules
import ground_truth_model
import scene


# Pinhole Camera Implementation
class PinholeCamera:
    """
    Pinhole camera model with rotation for camera measurements.

    Coordinate System Convention:
    - World coordinates: Standard XYZ with Z=height/up
    - Camera coordinates: Camera-centered with camera looking along camera's -Z axis
    - Points in front of camera have negative Z in camera coordinates

    The coordinate transform follows these steps:
    1. Translate points from world coordinates to camera-centered coordinates
    2. Apply rotation to align with camera's coordinate system
    3. Project 3D points onto 2D image plane using perspective projection

    Key equations:
    - P_cam = R × (P_world − C)
    - u = f × (x_cam / −z_cam) + u₀
    - v = f × (y_cam / −z_cam) + v₀

    Where:
    - C: Camera center in world coordinates
    - R: Rotation matrix from world to camera
    - f: Focal length (in pixels)
    - (u₀, v₀): Principal point (image center)
    """

    def __init__(self, position, rotation_angles, focal_length=800,
                 image_size=(1280, 960)):
        """
        Initialize a pinhole camera model.

        Args:
            position (list): [x, y, z] camera center in world coordinates
            rotation_angles (list): [roll, pitch, yaw] in radians. (rotating camera in fixed world frame)
                - roll: rotation around X-axis
                - pitch: rotation around Y-axis 
                - yaw: rotation around Z-axis 
            focal_length (float): Camera focal length in pixels
                - Controls zoom/FOV (higher values = more zoom, narrower FOV)
            image_size (tuple): (width, height) in pixels
        """
        self.C = np.array(position).reshape(3, 1)  # Camera center
        self.f = focal_length
        self.image_size = image_size
        self.u0, self.v0 = image_size[0] / 2, image_size[
            1] / 2  # Principal point

        # Build rotation matrix from Euler angles
        roll, pitch, yaw = rotation_angles

        # Individual rotation matrices
        Rx = np.array([  # Roll (around X)
            [1, 0, 0],
            [0, np.cos(-roll), -np.sin(-roll)],
            [0, np.sin(-roll), np.cos(-roll)]
        ])

        Ry = np.array([  # Pitch (around Y)
            [np.cos(-pitch), 0, np.sin(-pitch)],
            [0, 1, 0],
            [-np.sin(-pitch), 0, np.cos(-pitch)]
        ])

        Rz = np.array([  # Yaw (around Z)
            [np.cos(-yaw), -np.sin(-yaw), 0],
            [np.sin(-yaw), np.cos(-yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        # Note: visualize as rotating camera in fixed world frame
        self.R = Rz @ Ry @ Rx

    def world_to_camera(self, world_coords):
        """
        Transform point from world coordinates to camera coordinates.

        Args:
            world_coords: [x, y, z] position in world coordinates

        Returns:
            camera_coords: [x_cam, y_cam, z_cam] position in camera coordinates
        """
        # Extract position coordinates if given state vector
        if len(world_coords) > 3:
            position = world_coords[:3]
        else:
            position = world_coords

        # Convert to column vector
        p_world = np.array(position).reshape(3, 1)

        # Transform: p_cam = R(p_world - C)
        p_cam = self.R @ (p_world - self.C)

        return p_cam

    def is_in_front(self, camera_coords):
        """Check if point is in front of camera (z_cam < 0)"""
        return camera_coords[2, 0] < 0

    def is_in_frame(self, pixel_coords):
        """Check if pixel coordinates are within image bounds"""
        u, v = pixel_coords
        return (0 <= u < self.image_size[0] and 0 <= v < self.image_size[1])

    def project(self, camera_coords):
        """
        Apply perspective projection to camera coordinates.

        Args:
            camera_coords: 3D point in camera coordinates

        Returns:
            pixel_coords: [u, v] pixel coordinates
        """
        # Extract coordinates
        x_cam = camera_coords[0, 0]
        y_cam = camera_coords[1, 0]
        z_cam = camera_coords[2, 0]

        # Perspective projection (note the negative z)
        # Since camera looks along -Z, we use -z_cam in denominator
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        if abs(z_cam) < epsilon:
            # Point is essentially at the camera center, can't project meaningfully
            u, v = float('inf'), float('inf')
        else:
            u = self.f * x_cam / (-z_cam) + self.u0
            v = self.f * y_cam / (-z_cam) + self.v0

        return np.array([u, v])

    def g(self, world_coords):
        """
        Measurement function: Maps 3D world coordinates to 2D pixel coordinates.

        Args:
            world_coords: [x, y, z] position in world coordinates,
                          or [x, y, z, vx, vy, vz] state vector

        Returns:
            pixel_coords: [u, v] pixel coordinates
            is_visible: bool, whether point is visible to camera
        """
        # Transform to camera coordinates
        p_cam = self.world_to_camera(world_coords)

        # Check if point is in front of camera
        if not self.is_in_front(p_cam):
            return np.array([0, 0]), False

        # Project to image plane
        pixel = self.project(p_cam)

        # Check if projection is within image bounds
        is_visible = self.is_in_frame(pixel)

        return pixel, is_visible

    def jacobian(self, world_coords):
        """
        Calculate Jacobian of measurement function for EKF.

        Args:
            world_coords: [x, y, z] position in world coordinates

        Returns:
            H: 2x3 Jacobian matrix of measurement function
            is_visible: bool, whether point is visible to camera
        """
        # Transform to camera coordinates
        p_cam = self.world_to_camera(world_coords)

        # Check if point is in front of camera
        if not self.is_in_front(p_cam):
            return None, False

        # Extract coordinates
        x_cam = p_cam[0, 0]
        y_cam = p_cam[1, 0]
        z_cam = p_cam[2, 0]

        # Compute Jacobian for perspective projection
        # ∂u/∂x_cam = f/(-z_cam)
        # ∂u/∂y_cam = 0
        # ∂u/∂z_cam = f*x_cam/z_cam²
        # ∂v/∂x_cam = 0
        # ∂v/∂y_cam = f/(-z_cam)
        # ∂v/∂z_cam = f*y_cam/z_cam²
        J_proj = np.array([
            [self.f / (-z_cam), 0, self.f * x_cam / (z_cam * z_cam)],
            [0, self.f / (-z_cam), self.f * y_cam / (z_cam * z_cam)]
        ])

        # Jacobian of camera transform is just the rotation matrix
        J_cam = self.R

        # Full Jacobian is composition: H = J_proj @ J_cam
        H = J_proj @ J_cam

        # Check if projection is within image bounds
        pixel = self.project(p_cam)
        is_visible = self.is_in_frame(pixel)

        return H, is_visible


# Function to extract measurements for estimation algorithms
def get_camera_measurements(cameras, trajectory, noise_std=1.0):
    """
    Generate camera measurements with noise for use in estimation algorithms.

    Args:
        cameras: List of PinholeCamera objects
        trajectory: Nx3 array of trajectory positions
        noise_std: Standard deviation of measurement noise in pixels

    Returns:
        measurements: List of measurement arrays per camera
        visibilities: List of boolean arrays indicating visible points
    """
    all_measurements = []
    all_visibilities = []

    for cam in cameras:
        camera_measurements = []
        camera_visibilities = []

        for point in trajectory:
            # Extract position coordinates if needed
            world_coords = point[:3] if len(point) > 3 else point
            pixel, visible = cam.g(world_coords)

            if visible:
                # Add Gaussian noise to measurements
                noisy_pixel = pixel + np.random.normal(0, noise_std, size=2)
                camera_measurements.append(noisy_pixel)
            else:
                # For invisible points, add NaN values
                camera_measurements.append(np.array([np.nan, np.nan]))

            camera_visibilities.append(visible)

        all_measurements.append(np.array(camera_measurements))
        all_visibilities.append(np.array(camera_visibilities))

    return np.asarray(all_measurements), np.asarray(all_visibilities)
