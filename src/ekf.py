import numpy as np
import constants as consts


class EKF:
    """
    EKF Estimation Algorithm for tennis ball tracking
    """

    def __init__(self, mu_initial, sigma_initial, Q, R, dt):
        """
        Initialize the EKF object

        :param mu_initial: initial guess for state vector
        :param sigma_initial: initial guess for covariance matrix
        :param Q: process noise covariance matrix
        :param R: measurement noise covariance matrix
        :param dt: time step
        """
        self.mu = mu_initial
        self.sigma = sigma_initial
        self.Q = Q
        self.R = R
        self.dt = dt

    def set_mu(self, mu):
        self.mu = mu

    def set_sigma(self, sigma):
        self.sigma = sigma

    def f(self, state):
        """
        Compute state dynamics. In this case, the

        :param state: 6-dimensional state vector (3 position
        states, 3 velocity states): [x, y, z, xdot, ydot, zdot]
        """
        x, y, z, xdot, ydot, zdot = state
        x += self.dt * xdot
        y += self.dt * ydot
        z += self.dt * zdot + 0.5 * consts.g * (self.dt ** 2)
        zdot -= consts.g * self.dt
        return np.array([x, y, z, xdot, ydot, zdot])

    def A(self):
        """
        Compute the linearized state dynamics
        :return: 6x6 matrix
        """
        A = np.eye(6)
        A[0:3, 3:] = self.dt * np.eye(3)
        return A

    def predict(self, mu, sigma):
        """
        Run the prediction step of the EKF algorithm.

        :param mu: mean state vector
        :param sigma: state covariance matrix
        :return: the mean and covariance predictions
        """
        A = self.A()
        mu_predict = self.f(mu)
        sigma_predict = A @ sigma @ A.T + self.Q
        return mu_predict, sigma_predict

    def update(self, mu, sigma, cameras, y):
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
        assert len(cameras) > 0, "no cameras provided"
        assert len(cameras) == (np.shape(y)[0]), ("number of cameras does not "
                                                  "match number of "
                                                  "measurements")

        y_est = np.zeros_like(y)
        C = np.zeros((np.shape(y)[0] * 2, 6))
        for i, cam in enumerate(cameras):
            pixel, is_visible = cam.g(mu)
            y_est[i, :] = pixel
            H, is_visible = cam.jacobian(mu)
            C[2 * i: 2 * i + 2, :3] = H
        K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + self.R)
        mu_update = mu + (K @ np.array([(y - y_est).flatten()]).T).flatten()
        sigma_update = sigma - K @ C @ sigma
        return mu_update, sigma_update

    def run(self, cameras, y):
        mu_hist = []
        sigma_hist = []
        for i in range(np.shape(y)[1]):
            mu_predict, sigma_predict = self.predict(self.mu, self.sigma)
            self.mu, self.sigma = self.update(mu_predict, sigma_predict,
                                              cameras, y[:, i, :])
            mu_hist.append(self.mu)
            sigma_hist.append(self.sigma)
        return np.asarray(mu_hist), np.asarray(sigma_hist)
