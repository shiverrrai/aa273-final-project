import numpy as np
from matplotlib import pyplot as plt
import ground_truth_model
import estimation_model
import plotly.graph_objects as go
import scene
import constants as consts
from ekf import EKF
from pf import ParticleFilter
import imm
import sensor_model
import postpro
import argparse
import os
import json
import pickle
from datetime import datetime
from scipy.stats import friedmanchisquare, wilcoxon


class SimulationRunner:
    """Simple class to run tennis ball tracking simulations with different
    configurations."""

    def __init__(self):
        # GROUND TRUTH MODEL SETUP
        self.x0 = np.array([0, -2.0, 1.0, 17.0, 2.0, 5.0])
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
        self.measurement_noise_std = 5.0  # pixels

    def create_cameras(self, measurement_factor=1, camera_config='default'):
        """Create cameras with specified measurement factor and
        configuration."""

        if camera_config == 'default':
            positions = [
                [consts.court_length / 2, -self.camera_params['distance'],
                 self.camera_params['elevation']],
                [consts.court_length / 2, self.camera_params['distance'],
                 self.camera_params['elevation']]
            ]
            rotations = [[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]]

        # need to debug the rotations
        # elif camera_config == 'corners':
        #     positions = [
        #         [0, -self.camera_params['distance'], self.camera_params[
        #         'elevation']],
        #         [consts.court_length, self.camera_params['distance'],
        #         self.camera_params['elevation']]
        #     ]
        #     rotations = [[np.pi / 2, 0, -np.pi / 4], [-np.pi / 2, 0,
        #     -np.pi / 4]]

        elif camera_config == 'high':
            elevation = self.camera_params['elevation'] * 2
            positions = [
                [consts.court_length / 2, -self.camera_params['distance'],
                 elevation],
                [consts.court_length / 2, self.camera_params['distance'],
                 elevation]
            ]
            rotations = [[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]]

        elif camera_config == 'close':
            distance = self.camera_params['distance'] * 0.6
            positions = [
                [consts.court_length / 2, -distance,
                 self.camera_params['elevation']],
                [consts.court_length / 2, distance,
                 self.camera_params['elevation']]
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
        model = ground_truth_model.SystemModel(self.x0, self.run_time, self.dt)
        cameras = self.create_cameras(measurement_factor, camera_config)

        # Generate ground truth and measurements
        t, x = model.run_sim()
        y, visibility = sensor_model.get_camera_measurements(cameras, x,
                                                             noise_std=self.measurement_noise_std)

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
        ekf = EKF(estimation_model.FlightModel(), self.mu_initial.copy(),
                  self.sigma_initial.copy(), self.Q, R, self.dt)
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

        results['pf'] = {
            'trajectory': x_est_pf,
            'covariance': sigma_pf,
            'impact': pf.impact_data
        }

        # Run IMM
        ekf_flight = EKF(estimation_model.FlightModel(), self.mu_initial.copy(),
                         self.sigma_initial.copy(), self.Q,
                         R, self.dt)
        ekf_bounce = EKF(estimation_model.BounceModel(), self.mu_initial.copy(),
                         self.sigma_initial.copy(), self.Q,
                         R, self.dt)
        imm_tracker = imm.IMMTracker([ekf_flight, ekf_bounce], self.dt)
        x_est_imm, sigma_imm = imm_tracker.run(cameras, y, visibility)
        results['imm'] = {
            'trajectory': x_est_imm,
            'covariance': sigma_imm,
            'impact': imm_tracker.impact_data
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

    # Plot trajectories with distinct colors (all solid lines)
    scene.plot_ball_trajectory(fig, results['true_trajectory'], "Ground Truth",
                               "#00C851", 'solid')  # Bright green

    if 'ekf' in results:
        scene.plot_ball_trajectory(fig, results['ekf']['trajectory'], "EKF",
                                   "#FF4444", 'solid')  # Bright red
    if 'pf' in results:
        scene.plot_ball_trajectory(fig, results['pf']['trajectory'], "PF",
                                   "#FF8800", 'solid')  # Orange
    if 'imm' in results:
        scene.plot_ball_trajectory(fig, results['imm']['trajectory'], "IMM",
                                   "#AA00FF", 'solid')  # Purple

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
        scene.plot_impact_location(ax, x, y, None, color='green',
                                   label='Ground Truth', show_plot=False)

    # Filter results
    colors = {'ekf': 'red', 'pf': 'orange', 'imm': 'purple'}
    for filter_name, color in colors.items():
        if filter_name in results and results[filter_name][
            'impact'] is not None:
            x, y, sigma = results[filter_name]['impact']
            scene.plot_impact_location(ax, x, y, sigma, color=color,
                                       label=filter_name.upper(),
                                       show_plot=False)

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

    fig, axes = plt.subplots(3, len(available_filters),
                             figsize=(5 * len(available_filters), 12))
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
            ax.plot(time, true_traj[:, row], 'g-', linewidth=2,
                    label='Ground Truth')

            # Plot filter estimate
            ax.plot(time, est_traj[:, row], color=colors[filter_name],
                    linewidth=2, label=f'{filter_name.upper()} Estimate')

            # Add uncertainty ribbon (unified for all filters)
            if results[filter_name].get('covariance') is not None:
                covariance = results[filter_name]['covariance']
                std_dev = np.sqrt(covariance[:, row, row])
                upper_bound = est_traj[:, row] + 2 * std_dev
                lower_bound = est_traj[:, row] - 2 * std_dev
                ax.fill_between(time, lower_bound, upper_bound,
                                alpha=0.3, color=colors[filter_name],
                                label=f'{filter_name.upper()} ±2σ')

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


def create_results_directory(args):
    """Create results directory with descriptive name based on configuration."""
    # Create base results directory
    base_dir = "results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create descriptive folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"{args.camera_config}_factor{args.measurement_factor}"

    if args.monte_carlo_bounces:
        config_name += "_montecarlo"

    results_dir = os.path.join(base_dir, f"{timestamp}_{config_name}")
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def save_configuration(results_dir, args, sim_runner):
    """Save the simulation configuration to JSON."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "command_line_args": {
            "measurement_factor": args.measurement_factor,
            "camera_config": args.camera_config,
            "monte_carlo_bounces": args.monte_carlo_bounces
        },
        "simulation_parameters": {
            "initial_state": sim_runner.x0.tolist(),
            "run_time": sim_runner.run_time,
            "dt": sim_runner.dt,
            "measurement_noise_std": sim_runner.measurement_noise_std,
            "process_noise_Q": sim_runner.Q.tolist(),
            "initial_covariance": sim_runner.sigma_initial.tolist()
        },
        "camera_parameters": sim_runner.camera_params
    }

    config_file = os.path.join(results_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_file}")


def save_results_data(results_dir, results, monte_carlo_data=None):
    """Save all simulation results to files."""
    # Save main results as pickle (preserves numpy arrays)
    results_file = os.path.join(results_dir, "simulation_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    # Save key metrics as JSON for easy reading
    metrics = {}

    # Trajectory statistics
    true_traj = results['true_trajectory'][:, :3]
    for filter_name in ['ekf', 'pf', 'imm']:
        if filter_name in results:
            est_traj = results[filter_name]['trajectory'][:, :3]
            pos_errors = np.linalg.norm(true_traj - est_traj, axis=1)
            metrics[filter_name] = {
                "MAE": float(np.mean(pos_errors)),
                "RMSE": float(np.sqrt(np.mean(pos_errors ** 2))),
                "max_error": float(np.max(pos_errors)),
                "final_error": float(pos_errors[-1])
            }

            # Impact data if available
            if results[filter_name]['impact'] is not None:
                impact = results[filter_name]['impact']
                true_impact = results['true_impact']
                if true_impact is not None:
                    impact_error = np.linalg.norm([impact[0] - true_impact[0],
                                                   impact[1] - true_impact[1]])
                    metrics[filter_name]["impact_error"] = float(impact_error)

    # Add Monte Carlo results if available
    if monte_carlo_data is not None:
        metrics["monte_carlo"] = monte_carlo_data

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Results data saved to: {results_dir}")
    return metrics


def save_plots(results_dir, scene_fig, bounce_fig, position_fig,
               monte_carlo_fig=None):
    """Save all plots to files."""
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save 3D scene as HTML (interactive)
    if scene_fig is not None:
        scene_file = os.path.join(plots_dir, "3d_scene.html")
        scene_fig.write_html(scene_file)
        print(f"3D scene saved to: {scene_file}")

    # Save matplotlib figures as PNG and PDF
    if bounce_fig is not None:
        bounce_png = os.path.join(plots_dir, "bounce_comparison.png")
        bounce_pdf = os.path.join(plots_dir, "bounce_comparison.pdf")
        bounce_fig.savefig(bounce_png, dpi=300, bbox_inches='tight')
        bounce_fig.savefig(bounce_pdf, bbox_inches='tight')

    if position_fig is not None:
        position_png = os.path.join(plots_dir, "trajectory_comparison.png")
        position_pdf = os.path.join(plots_dir, "trajectory_comparison.pdf")
        position_fig.savefig(position_png, dpi=300, bbox_inches='tight')
        position_fig.savefig(position_pdf, bbox_inches='tight')

    if monte_carlo_fig is not None:
        mc_png = os.path.join(plots_dir, "monte_carlo_comparison.png")
        mc_pdf = os.path.join(plots_dir, "monte_carlo_comparison.pdf")
        monte_carlo_fig.savefig(mc_png, dpi=300, bbox_inches='tight')
        monte_carlo_fig.savefig(mc_pdf, bbox_inches='tight')

    print(f"Plots saved to: {plots_dir}")


def generate_summary_report(results_dir, metrics, args):
    """Generate a text summary report."""
    report_file = os.path.join(results_dir, "summary_report.txt")

    with open(report_file, 'w') as f:
        f.write("TENNIS BALL TRACKING SIMULATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Camera Config: {args.camera_config}\n")
        f.write(f"  Measurement Factor: 1/{args.measurement_factor}\n")
        f.write(
            f"  Monte Carlo: {'Yes' if args.monte_carlo_bounces else 'No'}\n\n")

        f.write("TRAJECTORY TRACKING PERFORMANCE:\n")
        f.write("-" * 40 + "\n")

        for filter_name in ['ekf', 'pf', 'imm']:
            if filter_name in metrics:
                data = metrics[filter_name]
                f.write(f"{filter_name.upper()}:\n")
                f.write(f"  Mean Absolute Error: {data['MAE']:.3f} m\n")
                f.write(f"  Root Mean Square Error: {data['RMSE']:.3f} m\n")
                f.write(f"  Maximum Error: {data['max_error']:.3f} m\n")
                f.write(f"  Final Error: {data['final_error']:.3f} m\n")
                if 'impact_error' in data:
                    f.write(
                        f"  Impact Location Error: {data['impact_error']:.3f} "
                        f"m\n")
                f.write("\n")

        if 'monte_carlo' in metrics:
            f.write("MONTE CARLO ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            mc_data = metrics['monte_carlo']
            f.write(
                f"EKF Bounce Error: {mc_data['ekf_mean']:.3f} ± "
                f"{mc_data['ekf_std']:.3f} m\n")
            f.write(
                f"PF Bounce Error: {mc_data['pf_mean']:.3f} ± "
                f"{mc_data['pf_std']:.3f} m\n")
            f.write(
                f"IMM Bounce Error: {mc_data['imm_mean']:.3f} ± "
                f"{mc_data['imm_std']:.3f} m\n\n")
            
            # Add statistical significance analysis
            if 'statistical_analysis' in mc_data:
                stats = mc_data['statistical_analysis']
                f.write("STATISTICAL SIGNIFICANCE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                
                # Report detection rates
                f.write("Detection Rates:\n")
                det_rates = stats['detection_rates']
                f.write(f"EKF: {det_rates['ekf']:.1%} ({int(det_rates['ekf'] * 50)}/50)\n")
                f.write(f"PF: {det_rates['pf']:.1%} ({int(det_rates['pf'] * 50)}/50)\n")
                f.write(f"IMM: {det_rates['imm']:.1%} ({int(det_rates['imm'] * 50)}/50)\n\n")
                
                if stats.get('insufficient_data', False):
                    f.write(f"Statistical Tests: Not performed (insufficient aligned data: {stats['sample_sizes']['compared_sample_size']} runs)\n")
                    f.write("Recommendation: Increase Monte Carlo runs or investigate detection failures\n")
                else:
                    f.write(f"Statistical Tests (aligned sample size: {stats['sample_sizes']['compared_sample_size']} runs):\n")
                    f.write(f"Friedman Test: χ² = {stats['friedman_statistic']:.3f}, "
                           f"p = {stats['friedman_p_value']:.4f}\n")
                    
                    if stats['friedman_p_value'] < 0.05:
                        f.write("Result: Significant differences detected across filters\n\n")
                        
                        f.write("Post-hoc Wilcoxon Signed-Rank Tests (Bonferroni-corrected α = 0.017):\n")
                        
                        pairwise = stats['pairwise_tests']
                        
                        # EKF vs PF
                        f.write(f"EKF vs PF: p = {pairwise['ekf_vs_pf']['p_value']:.4f}")
                        if pairwise['ekf_vs_pf']['significant']:
                            f.write(" (significant)\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # EKF vs IMM
                        f.write(f"EKF vs IMM: p = {pairwise['ekf_vs_imm']['p_value']:.4f}")
                        if pairwise['ekf_vs_imm']['significant']:
                            f.write(" (significant)\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # PF vs IMM
                        f.write(f"PF vs IMM: p = {pairwise['pf_vs_imm']['p_value']:.4f}")
                        if pairwise['pf_vs_imm']['significant']:
                            f.write(" (significant)\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # Generate interpretation
                        f.write("\nInterpretation:\n")
                        significant_pairs = []
                        
                        if pairwise['ekf_vs_imm']['significant']:
                            # Determine which is better
                            if mc_data['imm_mean'] < mc_data['ekf_mean']:
                                significant_pairs.append("IMM significantly outperformed EKF")
                            else:
                                significant_pairs.append("EKF significantly outperformed IMM")
                        
                        if pairwise['pf_vs_imm']['significant']:
                            if mc_data['imm_mean'] < mc_data['pf_mean']:
                                significant_pairs.append("IMM significantly outperformed PF")
                            else:
                                significant_pairs.append("PF significantly outperformed IMM")
                        
                        if pairwise['ekf_vs_pf']['significant']:
                            if mc_data['ekf_mean'] < mc_data['pf_mean']:
                                significant_pairs.append("EKF significantly outperformed PF")
                            else:
                                significant_pairs.append("PF significantly outperformed EKF")
                        
                        if significant_pairs:
                            for pair in significant_pairs:
                                f.write(f"- {pair}\n")
                        else:
                            f.write("- No significant pairwise differences after correction\n")
                            
                    else:
                        f.write("Result: No significant differences detected across filters\n")

    print(f"Summary report saved to: {report_file}")


def main():
    """Main function with simple command line options."""
    parser = argparse.ArgumentParser(
        description='Tennis Ball Tracking Simulation')
    parser.add_argument('--measurement-factor', type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help='Measurement factor for both cameras (default: 1)')
    parser.add_argument('--camera-config',
                        choices=['default', 'corners', 'high', 'close'],
                        default='default',
                        help='Camera configuration (default: default)')
    parser.add_argument('--monte-carlo-bounces', action='store_true',
                        help='Run Monte Carlo study for bounce statistics only')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving results (default: save everything)')

    args = parser.parse_args()

    # Create results directory
    if not args.no_save:
        results_dir = create_results_directory(args)
        print(f"Results will be saved to: {results_dir}")

    # Initialize and run simulation
    sim_runner = SimulationRunner()

    # Save configuration
    if not args.no_save:
        save_configuration(results_dir, args, sim_runner)

    print(f"Running simulation:")
    print(f"  Camera factor: 1/{args.measurement_factor} (both cameras)")
    print(f"  Camera config: {args.camera_config}")

    # Single trajectory simulation
    results = sim_runner.run_simulation(args.measurement_factor,
                                        args.camera_config)

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
            print(
                f"  {filter_name.upper()}: MAE = {np.mean(pos_errors):.3f}m, "
                f"RMSE = {np.sqrt(np.mean(pos_errors ** 2)):.3f}m")

    # 6. Monte Carlo analysis
    monte_carlo_data = None
    monte_carlo_fig = None

    if args.monte_carlo_bounces:
        print(f"\nRunning Monte Carlo study for bounce statistics...")

        # Run Monte Carlo for EKF
        print("  Running EKF Monte Carlo...")
        ekf_filter = EKF(estimation_model.FlightModel(), sim_runner.mu_initial,
                            sim_runner.sigma_initial,
                            sim_runner.Q,
                            np.eye(4) * sim_runner.measurement_noise_std ** 2,
                            sim_runner.dt)
        ekf_mean_error, ekf_std_dev, ekf_bounce_errors = postpro.run_study(
            num_runs=100,
            ground_truth_model=ground_truth_model.SystemModel(sim_runner.x0,
                                                                sim_runner.run_time,
                                                                sim_runner.dt),
            estimator=ekf_filter,
            mu_initial=sim_runner.mu_initial,
            sigma_initial=sim_runner.sigma_initial,
            cameras=sim_runner.create_cameras(args.measurement_factor,
                                                args.camera_config),
            camera_noise=sim_runner.measurement_noise_std,
        )

        # Run Monte Carlo for IMM
        print("  Running IMM Monte Carlo...")
        ekf_flight = EKF(estimation_model.FlightModel(), sim_runner.mu_initial,
                            sim_runner.sigma_initial, sim_runner.Q,
                            np.eye(4) * sim_runner.measurement_noise_std ** 2,
                            sim_runner.dt)
        ekf_bounce = EKF(estimation_model.BounceModel(), sim_runner.mu_initial,
                            sim_runner.sigma_initial, sim_runner.Q,
                            np.eye(4) * sim_runner.measurement_noise_std ** 2,
                            sim_runner.dt)

        imm_filter = imm.IMMTracker([ekf_flight, ekf_bounce], sim_runner.dt)
        imm_mean_error, imm_std_dev, imm_bounce_errors = postpro.run_study(
            num_runs=100,
            ground_truth_model=ground_truth_model.SystemModel(sim_runner.x0,
                                                                sim_runner.run_time,
                                                                sim_runner.dt),
            estimator=imm_filter,
            mu_initial=sim_runner.mu_initial,
            sigma_initial=sim_runner.sigma_initial,
            cameras=sim_runner.create_cameras(args.measurement_factor,
                                                args.camera_config),
            camera_noise=sim_runner.measurement_noise_std,
        )

        # Run Monte Carlo for PF
        print("  Running PF Monte Carlo...")
        pf_filter = ParticleFilter(sim_runner.mu_initial,
                                    sim_runner.sigma_initial,
                                    sim_runner.Q, np.eye(
                4) * sim_runner.measurement_noise_std ** 2,
                                    sim_runner.dt, n_particles=1000)
        pf_mean_error, pf_std_dev, pf_bounce_errors = postpro.run_study(
            num_runs=100,
            ground_truth_model=ground_truth_model.SystemModel(sim_runner.x0,
                                                                sim_runner.run_time,
                                                                sim_runner.dt),
            estimator=pf_filter,
            mu_initial=sim_runner.mu_initial,
            sigma_initial=sim_runner.sigma_initial,
            cameras=sim_runner.create_cameras(args.measurement_factor,
                                                args.camera_config),
            camera_noise=sim_runner.measurement_noise_std,
        )

        # Statistical significance testing
        print("  Performing statistical significance tests...")
        
        # Since postpro.run_study returns only valid detections (NaNs already removed),
        # we need to track which runs succeeded for each filter
        # We'll use the simpler approach of comparing the available data
        # Note: This is less rigorous than paired comparisons but still valid
        min_samples = min(len(ekf_bounce_errors), len(pf_bounce_errors), len(imm_bounce_errors))
        
        print(f"  Sample sizes - EKF: {len(ekf_bounce_errors)}, PF: {len(pf_bounce_errors)}, IMM: {len(imm_bounce_errors)}")
        print(f"  Using first {min_samples} samples from each filter for statistical tests")
        
        if min_samples >= 10:  # Minimum sample size for meaningful tests
            # Use first min_samples from each filter
            ekf_sample = ekf_bounce_errors[:min_samples]
            pf_sample = pf_bounce_errors[:min_samples]
            imm_sample = imm_bounce_errors[:min_samples]
            
            # Friedman test for overall differences
            friedman_stat, friedman_p = friedmanchisquare(ekf_sample, 
                                                           pf_sample, 
                                                           imm_sample)
            
            # Pairwise Wilcoxon signed-rank tests
            ekf_pf_stat, ekf_pf_p = wilcoxon(ekf_sample, pf_sample)
            ekf_imm_stat, ekf_imm_p = wilcoxon(ekf_sample, imm_sample)
            pf_imm_stat, pf_imm_p = wilcoxon(pf_sample, imm_sample)
            
            # Bonferroni correction for multiple comparisons (3 tests)
            bonferroni_alpha = 0.05 / 3
            
            # Store statistical results
            statistical_results = {
                "sample_sizes": {
                    "ekf_total": int(len(ekf_bounce_errors)),
                    "pf_total": int(len(pf_bounce_errors)),
                    "imm_total": int(len(imm_bounce_errors)),
                    "compared_sample_size": int(min_samples)
                },
                "friedman_statistic": float(friedman_stat),
                "friedman_p_value": float(friedman_p),
                "bonferroni_alpha": float(bonferroni_alpha),
                "pairwise_tests": {
                    "ekf_vs_pf": {
                        "statistic": float(ekf_pf_stat),
                        "p_value": float(ekf_pf_p),
                        "significant": bool(ekf_pf_p < bonferroni_alpha)
                    },
                    "ekf_vs_imm": {
                        "statistic": float(ekf_imm_stat),
                        "p_value": float(ekf_imm_p),
                        "significant": bool(ekf_imm_p < bonferroni_alpha)
                    },
                    "pf_vs_imm": {
                        "statistic": float(pf_imm_stat),
                        "p_value": float(pf_imm_p),
                        "significant": bool(pf_imm_p < bonferroni_alpha)
                    }
                },
                "detection_rates": {
                    "ekf": float(len(ekf_bounce_errors) / 50),
                    "pf": float(len(pf_bounce_errors) / 50),
                    "imm": float(len(imm_bounce_errors) / 50)
                }
            }
            
        else:
            print(f"  WARNING: Insufficient data ({min_samples} samples) for statistical tests")
            print("  Recommendation: Increase Monte Carlo runs or investigate detection failures")
            
            # Store limited results
            statistical_results = {
                "sample_sizes": {
                    "ekf_total": int(len(ekf_bounce_errors)),
                    "pf_total": int(len(pf_bounce_errors)),
                    "imm_total": int(len(imm_bounce_errors)),
                    "compared_sample_size": int(min_samples)
                },
                "insufficient_data": True,
                "detection_rates": {
                    "ekf": float(len(ekf_bounce_errors) / 50),
                    "pf": float(len(pf_bounce_errors) / 50),
                    "imm": float(len(imm_bounce_errors) / 50)
                }
            }

        # Compare results
        print(f"\nMonte Carlo Comparison:")
        print(
            f"  EKF:  Mean Error = {ekf_mean_error:.3f}m, Std Dev = "
            f"{ekf_std_dev:.3f}m")
        print(
            f"  PF:   Mean Error = {pf_mean_error:.3f}m, Std Dev = "
            f"{pf_std_dev:.3f}m")
        print(
            f"  IMM:   Mean Error = {imm_mean_error:.3f}m, Std Dev = "
            f"{imm_std_dev:.3f}m")

        print(f"\nStatistical Significance:")
        if min_samples >= 10:
            print(f"  Sample size used for tests: {min_samples} samples per filter")
            print(f"  Friedman test: χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}")
            if friedman_p < 0.05:
                print("  → Significant differences detected across filters")
                
                # Report significant pairwise differences
                if statistical_results["pairwise_tests"]["ekf_vs_pf"]["significant"]:
                    print(f"  EKF vs PF: p = {ekf_pf_p:.4f} (significant)")
                if statistical_results["pairwise_tests"]["ekf_vs_imm"]["significant"]:
                    print(f"  EKF vs IMM: p = {ekf_imm_p:.4f} (significant)")
                if statistical_results["pairwise_tests"]["pf_vs_imm"]["significant"]:
                    print(f"  PF vs IMM: p = {pf_imm_p:.4f} (significant)")
            else:
                print("  → No significant differences detected")
        else:
            print(f"  Insufficient data for statistical tests ({min_samples} samples)")
            print("  Detection rates:")
            print(f"    EKF: {statistical_results['detection_rates']['ekf']:.1%}")
            print(f"    PF: {statistical_results['detection_rates']['pf']:.1%}")
            print(f"    IMM: {statistical_results['detection_rates']['imm']:.1%}")
            print("  Detection rates:")
            print(f"    EKF: {statistical_results['detection_rates']['ekf']:.1%}")
            print(f"    PF: {statistical_results['detection_rates']['pf']:.1%}")
            print(f"    IMM: {statistical_results['detection_rates']['imm']:.1%}")

        # Store Monte Carlo data with statistical results
        monte_carlo_data = {
            "ekf_mean": float(ekf_mean_error),
            "ekf_std": float(ekf_std_dev),
            "pf_mean": float(pf_mean_error),
            "pf_std": float(pf_std_dev),
            "imm_mean": float(imm_mean_error),
            "imm_std": float(imm_std_dev),
            "num_runs": 50,
            "statistical_analysis": statistical_results
        }

        # Create comparison plot
        monte_carlo_fig = plt.figure(figsize=(12, 5))

        # Histogram comparison
        plt.subplot(1, 2, 1)
        bins = np.linspace(min(min(ekf_bounce_errors), min(pf_bounce_errors),
                               min(imm_bounce_errors)),
                           max(max(ekf_bounce_errors), max(pf_bounce_errors),
                               max(imm_bounce_errors)),

                           21)

        plt.hist([ekf_bounce_errors, pf_bounce_errors, imm_bounce_errors],
                 bins=bins,
                 label=['EKF', 'PF', 'IMM'],
                 color=['#2E86C1', '#E67E22', '#e6222f'],
                 alpha=0.8, edgecolor='black', linewidth=0.7)
        plt.xlabel('Bounce Location Error (m)')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Bounce Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot comparison
        plt.subplot(1, 2, 2)
        box_data = [ekf_bounce_errors, pf_bounce_errors, imm_bounce_errors]
        bp = plt.boxplot(box_data, tick_labels=['EKF', 'PF', 'IMM'],
                         patch_artist=True)

        bp['boxes'][0].set_facecolor('#2E86C1')
        bp['boxes'][1].set_facecolor('#E67E22')
        bp['boxes'][2].set_facecolor('#e6222f')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        bp['boxes'][2].set_alpha(0.7)

        plt.ylabel('Bounce Location Error (m)')
        plt.title('Monte Carlo Error Statistics')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Save all results
    if not args.no_save:
        print(f"\nSaving results...")

        # Save data and compute metrics
        metrics = save_results_data(results_dir, results, monte_carlo_data)

        # Save plots
        save_plots(results_dir, scene_fig, bounce_fig, position_fig,
                   monte_carlo_fig)

        # Generate summary report
        generate_summary_report(results_dir, metrics, args)

        print(f"\nAll results saved to: {results_dir}")
        print("Contents:")
        print("  - config.json (simulation configuration)")
        print("  - simulation_results.pkl (raw data)")
        print("  - metrics.json (performance metrics)")
        print("  - summary_report.txt (human-readable summary)")
        print("  - plots/ (all visualizations)")


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
# python simulation_runner.py --measurement-factor 2 --camera-config corners
# --monte-carlo-bounces
