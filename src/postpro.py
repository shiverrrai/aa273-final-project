import numpy as np
import sensor_model
from system_model import SystemModel


def run_study(num_runs: int, ground_truth_model: SystemModel,
              estimation_filter: any,
              mu_initial: np.array, sigma_initial: np.array, cameras: list):
    """
    Runs a comparative study between an estimation filter and a ground truth
    model. Computes a common set of metrics to assess filter performance.

    :param num_runs: number of runs to run the study for
    :param ground_truth_model: SystemModel instance
    :param estimation_filter: estimation filter instance (ie, EKF)
    :param mu_initial: initial estimation mean
    :param sigma_initial: initial estimation covariance
    :param cameras: list of PinholeCamera instances
    :return: mean error, standard deviation of error, percentage of missed
    detections
    """
    # reset objects and create containers
    ground_truth_bounces = []
    estimated_bounces = []
    ground_truth_model.reset()
    estimation_filter.reset(mu_initial, sigma_initial)

    # run study
    for i in range(num_runs):
        # compute ground truth trajectory
        t, x = ground_truth_model.run_sim()
        # compute ground truth bounce location
        gt_bounce_x, gt_bounce_y = ground_truth_model.x_impact[0], \
            ground_truth_model.x_impact[1]
        ground_truth_bounces.append([gt_bounce_x, gt_bounce_y])
        # compute measurement data for a given ground truth trajectory
        y, visibility = sensor_model.get_camera_measurements(cameras, x)
        # run estimation filter with given measurements and camera instances
        estimation_filter.run(cameras, y, visibility)
        # compute estimated bounce location (ignoring instances where a
        # bounce was failed to be detected)
        estimated_bounce_x, estimated_bounce_y = np.nan, np.nan
        if estimation_filter.impact_data is not None:
            estimated_bounce_x, estimated_bounce_y, ekf_bounce_sigma = (
                estimation_filter.impact_data)
        estimated_bounces.append([estimated_bounce_x, estimated_bounce_y])
        # reset objects before starting next run
        ground_truth_model.reset()
        estimation_filter.reset(mu_initial, sigma_initial)
    ground_truth_bounces = np.asarray(ground_truth_bounces)
    estimated_bounces = np.asarray(estimated_bounces)
    # compute error between ground truth and estimated bounce location
    error = ground_truth_bounces - estimated_bounces

    # count and store missed bounce detections
    missed_detections = np.sum(np.isnan(error[:, 0]))

    # drop missed bounce detections
    detected_bounce_errors = error[~np.isnan(error).any(axis=1), :]
    detected_bounce_errors = np.linalg.norm(detected_bounce_errors, axis=1)
    mean_error = np.mean(detected_bounce_errors)
    std_dev = np.std(detected_bounce_errors)
    print(f'Comparative study results: (n={num_runs})\n')
    print(f'Mean Bounce Location Error = {mean_error:.3f} m\n')
    print(f'Error Standard Deviation = {std_dev:.3f} m\n')
    print(f'Missed Detection Rate = {(missed_detections / num_runs) * 100:.1f}%\n')
    return mean_error, std_dev, detected_bounce_errors
