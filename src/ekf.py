import numpy as np

class EKF:
    """
    EKF Estimation Algorithm for tennis ball tracking
    """

    def __init__(self, model, mu_initial, sigma_initial, Q, R, dt):
        """
        Initialize the EKF object

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

        :param mu: mean state vector
        :param sigma: state covariance matrix
        :return: the mean and covariance predictions
        """
        A = self.model.A(self.dt)
        mu_predict = self.model.f(self.mu, self.dt)
        sigma_predict = A @ self.sigma @ A.T + self.Q
        return mu_predict, sigma_predict



    def update(self, cameras, y):
        """
        Runs the update step of the EKF algorithm.

        :param mu: the mean state vector
        :param sigma: the state covariance matrix
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
