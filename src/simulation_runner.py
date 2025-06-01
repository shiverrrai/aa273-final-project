import numpy as np
from matplotlib import pyplot as plt
import system_model
import plotly.graph_objects as go
import scene
import constants as consts
from ekf import EKF
from pf import ParticleFilter
import sensor_model
import postpro
import argparse

class SimulationRunner:
    """Simple class to run tennis ball tracking simulations with different configurations."""
    
    def __init__(self):
        # GROUND TRUTH MODEL SETUP
        self.x0 = np.array([0, 0, 1.0, 20.0, 0.0, 5.0])
        self.run_time = 10  # seconds
        self.dt = 0.01  # seconds
        
        # CAMERA MODEL SETUP
        self.camera_params = {
            'distance': 25,  # Meters from court centerline
            'elevation': 4,  # Meters above ground
            'focal_length': 1000,  # Pixels
            'image_size': (1280, 960),
        }
        
        # FILTER SETUP
        self.mu_initial = np.zeros(6)
        self.sigma_initial = np.eye(6)
        self.Q = 0.1 * np.eye(6)  # Process noise covariance
        self.measurement_noise_std = 1.0  # pixels
        
    def create_cameras(self, measurement_factor=1, camera_config='default'):
        """Create cameras with specified measurement factor and configuration."""
        
        if camera_config == 'default':
            positions = [
                [consts.court_length / 2, -self.camera_params['distance'], self.camera_params['elevation']],
                [consts.court_length / 2, self.camera_params['distance'], self.camera_params['elevation']]
            ]
            rotations = [[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]]
        
        # need to debug the rotations
        # elif camera_config == 'corners':
        #     positions = [
        #         [0, -self.camera_params['distance'], self.camera_params['elevation']],
        #         [consts.court_length, self.camera_params['distance'], self.camera_params['elevation']]
        #     ]
        #     rotations = [[np.pi / 2, 0, -np.pi / 4], [-np.pi / 2, 0, -np.pi / 4]]
            
        elif camera_config == 'high':
            elevation = self.camera_params['elevation'] * 2
            positions = [
                [consts.court_length / 2, -self.camera_params['distance'], elevation],
                [consts.court_length / 2, self.camera_params['distance'], elevation]
            ]
            rotations = [[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]]
            
        elif camera_config == 'close':
            distance = self.camera_params['distance'] * 0.6
            positions = [
                [consts.court_length / 2, -distance, self.camera_params['elevation']],
                [consts.court_length / 2, distance, self.camera_params['elevation']]
            ]
            rotations = [[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]]
        
        # Create cameras (both with same measurement factor)
        cameras = []
        for pos, rot in zip(positions, rotations):
            camera = sensor_model.PinholeCamera(
                position=pos,
                rotation_angles=rot,
                focal_length=self.camera_params['focal_length'],
                image_size=self.camera_params['image_size'],
                measurement_factor=measurement_factor
            )
            cameras.append(camera)
            
        return cameras
    
    def run_simulation(self, measurement_factor=1, camera_config='default'):
        """Run simulation with EKF, PF, and IMM filters."""
        
        # Create model and cameras
        model = system_model.SystemModel(self.x0, self.run_time, self.dt)
        cameras = self.create_cameras(measurement_factor, camera_config)
        
        # Generate ground truth and measurements
        t, x = model.run_sim()
        y, visibility = sensor_model.get_camera_measurements(cameras, x, noise_std=self.measurement_noise_std)
        
        # Measurement noise covariance (2 measurements per camera)
        R = np.eye(2 * len(cameras)) * (self.measurement_noise_std ** 2)
        
        # Initialize results
        results = {
            'time': t,
            'true_trajectory': x,
            'true_impact': model.x_impact,
            'cameras': cameras,
            'measurements': y,
            'visibility': visibility
        }
        
        # Run EKF
        ekf = EKF(self.mu_initial.copy(), self.sigma_initial.copy(), self.Q, R, self.dt)
        x_est_ekf, sigma_ekf = ekf.run(cameras, y, visibility)
        results['ekf'] = {
            'trajectory': x_est_ekf,
            'covariance': sigma_ekf,
            'impact': ekf.impact_data
        }

        # Run PF
        pf = ParticleFilter(self.mu_initial.copy(), self.sigma_initial.copy(), 
                           self.Q, R, self.dt, n_particles=1000)
        x_est_pf, sigma_pf = pf.run(cameras, y, visibility)
        
        # Get percentile bounds for uncertainty ribbons
        pf_lower, pf_upper = pf.get_percentile_bounds()
        
        results['pf'] = {
            'trajectory': x_est_pf,
            'covariance': sigma_pf,
            'impact': pf.impact_data,
            'particles': pf.particles_history,
            'percentile_bounds': (pf_lower, pf_upper)
        }
        
        # TODO: Add PF when implemented
        # pf = ParticleFilter(...)
        # x_est_pf = pf.run(cameras, y, visibility)
        # results['pf'] = {'trajectory': x_est_pf, 'impact': pf.impact_data}
        
        # TODO: Add IMM when implemented  
        # imm = IMM(...)
        # x_est_imm = imm.run(cameras, y, visibility)
        # results['imm'] = {'trajectory': x_est_imm, 'impact': imm.impact_data}
        
        return results


def plot_scene_results(results, title_suffix=""):
    """Plot 3D scene with trajectories and cameras."""
    fig = go.Figure()
    scene.draw_court(fig)
    
    # Plot trajectories
    scene.plot_ball_trajectory(fig, results['true_trajectory'], "Ground Truth", "green")
    
    if 'ekf' in results:
        scene.plot_ball_trajectory(fig, results['ekf']['trajectory'], "EKF", "red")
    if 'pf' in results:
        scene.plot_ball_trajectory(fig, results['pf']['trajectory'], "PF", "orange")
    if 'imm' in results:
        scene.plot_ball_trajectory(fig, results['imm']['trajectory'], "IMM", "purple")
    
    # Show cameras
    colors = ['red', 'blue']
    for i, cam in enumerate(results['cameras']):
        scene.draw_camera_frustum(
            fig, cam, colors[i], f"Camera {i + 1} (1/{cam.measurement_factor})",
            near_plane=5, far_plane=20
        )
    
    fig.update_layout(title=f'Tennis Ball Trajectory {title_suffix}')
    scene.show_scene(fig)
    return fig


def plot_bounce_results(results, title_suffix=""):
    """Plot bounce locations for all filters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ground truth
    if results['true_impact'] is not None:
        x, y = results['true_impact'][0], results['true_impact'][1]
        scene.plot_impact_location(ax, x, y, None, color='green', label='Ground Truth', show_plot=False)
    
    # Filter results
    colors = {'ekf': 'red', 'pf': 'orange', 'imm': 'purple'}
    for filter_name, color in colors.items():
        if filter_name in results and results[filter_name]['impact'] is not None:
            x, y, sigma = results[filter_name]['impact']
            scene.plot_impact_location(ax, x, y, sigma, color=color, 
                                     label=filter_name.upper(), show_plot=False)
    
    ax.set_title(f'Bounce Location Comparison {title_suffix}')
    ax.legend()
    plt.show()
    return fig


def plot_trajectory_comparison(results, title_suffix=""):
    """Plot X,Y,Z positions: each filter vs ground truth in separate columns."""
    available_filters = [f for f in ['ekf', 'pf', 'imm'] if f in results]
    
    if not available_filters:
        print("No filter results to plot")
        return None
    
    fig, axes = plt.subplots(3, len(available_filters), figsize=(5*len(available_filters), 12))
    if len(available_filters) == 1:
        axes = axes.reshape(-1, 1)  # Ensure 2D array
    
    time = results['time']
    true_traj = results['true_trajectory'][:, :3]
    colors = {'ekf': 'red', 'pf': 'orange', 'imm': 'purple'}
    components = ['X', 'Y', 'Z']
    
    for col, filter_name in enumerate(available_filters):
        est_traj = results[filter_name]['trajectory'][:, :3]
        
        for row, component in enumerate(components):
            ax = axes[row, col]
            
            # Plot ground truth
            ax.plot(time, true_traj[:, row], 'g-', linewidth=2, label='Ground Truth')
            
            # Plot filter estimate
            ax.plot(time, est_traj[:, row], color=colors[filter_name], 
                   linewidth=2, label=f'{filter_name.upper()} Estimate')
            
            # Add uncertainty ribbon
            if filter_name == 'ekf' and results[filter_name].get('covariance') is not None:
                covariance = results[filter_name]['covariance']
                std_dev = np.sqrt(covariance[:, row, row])
                upper_bound = est_traj[:, row] + 2 * std_dev
                lower_bound = est_traj[:, row] - 2 * std_dev
                ax.fill_between(time, lower_bound, upper_bound, 
                               alpha=0.3, color=colors[filter_name], 
                               label=f'±2σ')
            elif filter_name == 'pf' and results[filter_name].get('percentile_bounds') is not None:
                # PF: 2.5th to 97.5th percentile ribbon
                pf_lower, pf_upper = results[filter_name]['percentile_bounds']
                if pf_lower is not None:
                    ax.fill_between(time, pf_lower[:, row], pf_upper[:, row],
                                    alpha=0.3, color=colors[filter_name],
                                    label=f'{filter_name.upper()} 95% CI')
            
            # Format subplot
            ax.set_ylabel(f'{component} (m)')
            ax.set_title(f'{filter_name.upper()}: {component} Position')
            ax.legend()
            ax.grid(True)
            
            if row == 2:  # Bottom row
                ax.set_xlabel('Time (s)')
    
    plt.suptitle(f'Position Tracking Comparison {title_suffix}')
    plt.tight_layout()
    plt.show()
    return fig


def main():
    """Main function with simple command line options."""
    parser = argparse.ArgumentParser(description='Tennis Ball Tracking Simulation')
    parser.add_argument('--measurement-factor', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Measurement factor for both cameras (default: 1)')
    parser.add_argument('--camera-config', choices=['default', 'corners', 'high', 'close'], 
                       default='default', help='Camera configuration (default: default)')
    parser.add_argument('--monte-carlo-bounces', action='store_true',
                       help='Run Monte Carlo study for bounce statistics only')
    
    args = parser.parse_args()
    
    # Initialize and run simulation
    sim_runner = SimulationRunner()
    
    print(f"Running simulation:")
    print(f"  Camera factor: 1/{args.measurement_factor} (both cameras)")
    print(f"  Camera config: {args.camera_config}")
    
    # Single trajectory simulation
    results = sim_runner.run_simulation(args.measurement_factor, args.camera_config)
    
    # Create title suffix for plots
    title_suffix = f"({args.camera_config}, 1/{args.measurement_factor})"
    
    # Generate all plots
    print("Generating plots...")
    
    # 1. 3D Scene visualization
    scene_fig = plot_scene_results(results, title_suffix)
    scene_fig.show(renderer="browser")
    
    # 2. Bounce location comparison
    bounce_fig = plot_bounce_results(results, title_suffix)
    
    # 3. Position tracking comparison with uncertainty ribbons
    position_fig = plot_trajectory_comparison(results, title_suffix)
    
    # 4. Camera view images
    print("Generating camera views...")
    scene.generate_camera_views(results['cameras'], results['true_trajectory'])
    
    # 5. Print trajectory statistics
    print(f"\nTrajectory Statistics {title_suffix}:")
    true_traj = results['true_trajectory'][:, :3]
    
    for filter_name in ['ekf', 'pf', 'imm']:
        if filter_name in results:
            est_traj = results[filter_name]['trajectory'][:, :3]
            pos_errors = np.linalg.norm(true_traj - est_traj, axis=1)
            print(f"  {filter_name.upper()}: MAE = {np.mean(pos_errors):.3f}m, RMSE = {np.sqrt(np.mean(pos_errors**2)):.3f}m")
    
    # 6. Optional Monte Carlo for bounce statistics (postpro.py module)
    if args.monte_carlo_bounces:
        print(f"\nRunning Monte Carlo study for bounce statistics...")
        # Use existing postpro module
        mean_error, std_dev, bounce_errors = postpro.run_study(
            num_runs=100,
            ground_truth_model=system_model.SystemModel(sim_runner.x0, sim_runner.run_time, sim_runner.dt),
            estimation_filter=EKF(sim_runner.mu_initial, sim_runner.sigma_initial, 
                                sim_runner.Q, np.eye(4) * sim_runner.measurement_noise_std**2, sim_runner.dt),
            mu_initial=sim_runner.mu_initial,
            sigma_initial=sim_runner.sigma_initial,
            cameras=sim_runner.create_cameras(args.measurement_factor, args.camera_config)
        )

        # TODO do this for PF and IMM when implemented


if __name__ == "__main__":
    main()


# Example usage:

# # Default run
# python simulation_runner.py

# # Half-rate cameras  
# python simulation_runner.py --measurement-factor 2

# # Different camera positions
# python simulation_runner.py --camera-config high

# # With bounce statistics
# python simulation_runner.py --monte-carlo-bounces

# # Combined
# python simulation_runner.py --measurement-factor 2 --camera-config corners --monte-carlo-bounces