import numpy as np
import estimation_helpers as eh
import constants as consts

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
    position_window = [consts.court_length / 2, consts.court_length - buffer]
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

class EKF:
    """
    EKF Estimation Algorithm for tennis ball tracking
    """

    def __init__(self, model, mu_initial, sigma_initial, Q, R, dt):
        """
        Initialize the EKF object

        :param model: the dynamics model to use
        :param mu_initial: initial guess for state vector
        :param sigma_initial: initial guess for covariance matrix
        :param Q: process noise covariance matrix
        :param R: measurement noise covariance matrix
        :param dt: time step
        """
        self.model = model
        self.mu = mu_initial
        self.sigma = sigma_initial
        self.Q = Q
        self.R = R
        self.dt = dt
        self.impact_data = None
        self.likelihood = 0

    def set_state(self, mu, sigma):
        self.mu = mu.copy()
        self.sigma = sigma.copy()

    def get_state(self):
        return self.mu.copy(), self.sigma.copy()

    def reset(self, mu_initial, sigma_initial):
        self.mu = mu_initial
        self.sigma = sigma_initial
        self.impact_data = None

    def predict(self):
        """
        Run the prediction step of the EKF algorithm.

        :return: the mean and covariance predictions
        """
        A = self.model.A(self.dt)
        mu_predict = self.model.f(self.mu, self.dt)
        sigma_predict = A @ self.sigma @ A.T + self.Q
        return mu_predict, sigma_predict



    def update(self, cameras, y):
        """
        Runs the update step of the EKF algorithm.

        :param cameras: a list of PinholeCamera objects
        :param y: a (2m, ) measurement vector (where m is number of
        measurements. In this case, each measurement will
        be a (u,v) location in an image in pixels
        :return: the mean and covariance updates for the current time step
        """

        # if no cameras are provided, return the unmodified mean and covariances
        if len(cameras) == 0:
            return self.mu, self.sigma

        assert len(cameras) == (np.shape(y)[0]), ("number of cameras does not "
                                                  "match number of "
                                                  "measurements")

        y_est = np.zeros_like(y)
        C = np.zeros((np.shape(y)[0] * 2, 6))
        for i, cam in enumerate(cameras):
            pixel, is_visible = cam.g(self.mu)
            y_est[i, :] = pixel
            H, is_visible = cam.jacobian(self.mu)
            C[2 * i: 2 * i + 2, :3] = H
        S = C @ self.sigma @ C.T + self.R
        K = self.sigma @ C.T @ np.linalg.inv(S)
        phi = np.array([(y - y_est).flatten()])  # measurement innovation
        mu_update = self.mu + (K @ phi.T).flatten()
        sigma_update = self.sigma - K @ C @ self.sigma
        self.likelihood = np.exp(-0.5 * phi @ np.linalg.inv(S) @ phi.T) / np.sqrt(
            np.linalg.det(2 * np.pi * S))
        return mu_update, sigma_update

    def run(self, cameras, y, visibility):
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
            prev_mu, prev_sigma = self.get_state()
            mu_predict, sigma_predict = self.predict()
            self.set_state(mu_predict, sigma_predict)
            valid_cameras, visible_measurements = eh.bundle_visible_measurements(
                cameras, y[:, i, :], visibility[:, i])
            mu_update, sigma_update = self.update(valid_cameras,
                                                       visible_measurements)
            self.set_state(mu_update, sigma_update)
            x_star = check_ground_impact(prev_mu, prev_sigma, self.mu,
                                         self.sigma, self.dt)
            if x_star is not None and self.impact_data is None:
                self.impact_data = x_star
            mu_hist.append(self.mu)
            sigma_hist.append(self.sigma)
        return np.asarray(mu_hist), np.asarray(sigma_hist)
