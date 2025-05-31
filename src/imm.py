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


class FlightModel:
    def __init__(self, dt):
        self.dt = dt

    def f(self, state):
        """
        Compute state dynamics for ball in flight.

        :param state: 6-dimensional state vector (3 position
        states, 3 velocity states): [x, y, z, xdot, ydot, zdot]˜
        """
        x, y, z, xdot, ydot, zdot = state
        x += self.dt * xdot
        y += self.dt * ydot
        z += self.dt * zdot + 0.5 * consts.g * (self.dt ** 2)
        zdot += consts.g * self.dt
        return np.array([x, y, z, xdot, ydot, zdot])

    def A(self):
        """
        Compute the linearized state dynamics of ball in flight.
        :return: 6x6 matrix
        """
        A = np.eye(6)
        A[0:3, 3:] = self.dt * np.eye(3)
        return A


class BounceModel:
    def __init__(self, dt):
        self.dt = dt

    def f(self, state):
        """
        Compute state dynamics for ball during bounce.

        :param state: 6-dimensional state vector (3 position
        states, 3 velocity states): [x, y, z, xdot, ydot, zdot]˜
        """
        x, y, z, xdot, ydot, zdot = state
        x += self.dt * xdot
        y += self.dt * ydot
        z += self.dt * zdot + 0.5 * consts.g * (self.dt ** 2)
        zdot = -consts.e * zdot + consts.g * self.dt
        return np.array([x, y, z, xdot, ydot, zdot])

    def A(self):
        """
        Compute the linearized state dynamics of ball during bounce.
        :return: 6x6 matrix
        """
        A = np.eye(6)
        A[0:3, 3:] = self.dt * np.eye(3)
        A[5, 5] *= -consts.e
        return A


class GenericEkf:
    def __init__(self, model, mu_initial, sigma_initial, Q, R, dt):
        self.model = model
        self.mu = mu_initial
        self.sigma = sigma_initial
        self.Q = Q
        self.R = R
        self.dt = dt

    def predict(self):
        """
        Run the prediction step of the EKF algorithm.
        """
        A = self.model.A()
        self.mu = self.model.f(self.mu)
        self.sigma = A @ self.sigma @ A.T + self.Q
        return

    def update(self, cameras, y):
        """
        Runs the update step of the EKF algorithm.
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
        self.mu += (K @ phi.T).flatten()
        self.sigma = self.sigma - K @ C @ self.sigma
        likelihood = np.exp(-0.5 * phi @ np.linalg.inv(S) @ phi.T) / np.sqrt(
            np.linalg.det(2 * np.pi * S))
        return likelihood

    def set_state(self, mu, sigma):
        self.mu = mu.copy()
        self.sigma = sigma.copy()

    def get_state(self):
        return self.mu.copy(), self.sigma.copy()


class IMMTracker:
    def __init__(self, models, mu_initial, sigma_initial, transition_probs, dt):
        self.models = models
        self.P_ij = transition_probs
        # conditional model probabilities
        self.alpha = np.ones(len(models)) / len(models)
        self.dt = dt
        self.mu_fused = mu_initial
        self.sigma_fused = sigma_initial
        self.impact_data = None

    def reset(self, mu_initial, sigma_initial):
        self.mu_fused = mu_initial
        self.sigma_fused = sigma_initial
        self.impact_data = None

    def mix_states(self):
        N = len(self.models)
        mixed_x, mixed_P = [], []

        c_j = np.zeros(N)
        for j in range(N):
            c_j[j] = sum(self.P_ij[i, j] * self.alpha[i] for i in range(N))

        for j in range(N):
            x_j = np.zeros(6)
            P_j = np.zeros((6, 6))
            for i in range(N):
                x_i, P_i = self.models[i].get_state()
                weight = self.P_ij[i, j] * self.alpha[i] / c_j[j]
                dx = np.array([x_i - x_i.mean()]).T
                x_j += weight * x_i
                P_j += weight * (P_i + dx @ dx.T)
            mixed_x.append(x_j)
            mixed_P.append(P_j)
        return mixed_x, mixed_P, c_j

    def predict(self):
        mixed_x, mixed_P, _ = self.mix_states()
        for i, model in enumerate(self.models):
            model.set_state(mixed_x[i], mixed_P[i])
            model.predict()

    def update(self, cameras, y):
        # if no cameras are provided, return the unmodified mean and covariances
        if len(cameras) == 0:
            return self.mu_fused, self.sigma_fused

        likelihoods = [model.update(cameras, y) for model in self.models]
        c = np.zeros(len(self.models))
        for j in range(len(self.models)):
            c[j] = likelihoods[j] * sum(
                self.P_ij[i, j] * self.alpha[i] for i in
                range(len(self.models)))
        self.alpha = c / np.sum(c)
        mu_fused = sum(
            self.alpha[i] * self.models[i].mu for i in range(len(self.models)))
        sigma_fused = sum(
            self.alpha[i] * (self.models[i].sigma +
                             np.outer(self.models[i].mu - mu_fused,
                                      self.models[i].mu - mu_fused))
            for i in range(len(self.models))
        )
        return mu_fused, sigma_fused

    def run(self, cameras, y, visibility):
        mu_hist = []
        sigma_hist = []
        for i in range(np.shape(y)[1]):
            prev_mu, prev_sigma = self.mu_fused, self.sigma_fused
            self.predict()
            valid_cameras, visible_measurements = bundle_visible_measurements(
                cameras, y[:, i, :], visibility[:, i])
            self.mu_fused, self.sigma_fused = self.update(valid_cameras,
                                                          visible_measurements)
            x_star = check_ground_impact(prev_mu,
                                         prev_sigma, self.mu_fused,
                                         self.sigma_fused, self.dt)
            if x_star is not None and self.impact_data is None:
                self.impact_data = x_star
            mu_hist.append(self.mu_fused)
            sigma_hist.append(self.sigma_fused)
        return np.asarray(mu_hist), np.asarray(sigma_hist)
