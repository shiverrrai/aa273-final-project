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


def check_ground_impact(prev_state, prev_P, curr_state, curr_P, dt):
    """
    Checks whether the estimated state has made impact with the ground.
    Performs linear interpolation between previous and current state to
    compute state estimate results at moment of impact.

    :param prev_state: previous state estimate, (6,) np array
    :param prev_P: previous covariance matrix, (6,6) nd array
    :param curr_state: current state estimate, (6,) np array
    :param curr_P: current covariance matrix, (6,6) nd array
    :param dt: time step
    :return: np array of x & y location, along with covariance matrix at
    moment of impact
    """
    z0, z1 = prev_state[2], curr_state[2]
    x0, x1 = prev_state[0], curr_state[0]

    # search for bounce in a reasonable position window
    buffer = 1
    position_window = [consts.court_length / 2, consts.court_length + buffer]
    if position_window[0] <= x1 <= position_window[1] and (
            np.sign(z0) <= 0 or np.sign(z1) <= 0):
        alpha = z0 / (z0 - z1)
        t_impact = alpha * dt
        x_impact = prev_state[0] + alpha * (curr_state[0] - prev_state[0])
        y_impact = prev_state[1] + alpha * (curr_state[1] - prev_state[1])

        P0_xyz = prev_P[0:3, 0:3]
        P1_xyz = curr_P[0:3, 0:3]
        P_impact = (1 - alpha) * P0_xyz + alpha * P1_xyz
        sigma_xy = P_impact[0:2, 0:2]

        return np.array([x_impact, y_impact, sigma_xy], dtype=object)
    return None


def run_estimator(estimator, cameras, y, visibility):
    """
    Runs the EKF algorithm on time series measurement data. Performs
    prediction steps, update steps, and computes and populates impact
    location of state estimate. Note that this function performs the
    update step of the EKF algorithm only when a given measurement sample
    is visible.

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
