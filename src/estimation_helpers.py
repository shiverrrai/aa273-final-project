import numpy as np
import constants as consts


def bundle_visible_measurements(cameras, y, visiblity):
    """
    Bundles the visible (non-Nan) measurements received at a given time step.
    :param cameras: list of PinholeCamera objects (length m)
    :param y: (mx2) ndarray of measurements
    :param visiblity: (mx1) ndarray of bools representing whether
    measurements are visible
    :return:
    """
    valid_cameras = []
    visible_measurements = []
    for i in range(len(cameras)):
        if visiblity[i]:
            visible_measurements.append(y[i, :])
            valid_cameras.append(cameras[i])
    return valid_cameras, np.asarray(visible_measurements)


def run_estimator(estimator, cameras, y, visibility):
    """
    Runs the EKF algorithm on time series measurement data. Performs
    prediction steps, update steps, and computes and populates impact
    location of state estimate. Note that this function performs the
    update step of the EKF algorithm only when a given measurement sample
    is visible.

    :param estimator: EKF or IMM estimator object
    :param cameras: list of PinholeCamera objects
    :param y: (m, n, 2) measurement nd array where m is number of cameras,
    n is number of measurements, and 2 is the number of pixels per
    measurement (always 2 since it is a 2d image)
    :param visibility: (m, n) nd array of bools representing if the ball
    is visible or not for a given measurement.
    :return: (n, 6) array of mean estimates and (n, 6, 6) array of
    cavariance matrices. Also populates state and covariance matrix when
    ball makes impact with the ground.
    """
    mu_hist = []
    sigma_hist = []
    for i in range(np.shape(y)[1]):
        prev_mu, prev_sigma = estimator.get_state()
        mu_predict, sigma_predict = estimator.predict()
        estimator.set_state(mu_predict, sigma_predict)
        valid_cameras, visible_measurements = bundle_visible_measurements(
            cameras, y[:, i, :], visibility[:, i])
        mu_update, sigma_update = estimator.update(valid_cameras,
                                                         visible_measurements)
        estimator.set_state(mu_update, sigma_update)
        x_star = check_ground_impact(prev_mu, prev_sigma, estimator.mu,
                                     estimator.sigma, estimator.dt)
        if x_star is not None and estimator.impact_data is None:
            estimator.impact_data = x_star
        mu_hist.append(estimator.mu)
        sigma_hist.append(estimator.sigma)
    return np.asarray(mu_hist), np.asarray(sigma_hist)
