import numpy as np


def compute_transition_matrix(z, threshold=0.5, sharpness=10):
    """
    Adjust transition probabilities based on vertical position z.
    Returns a 2x2 transition matrix.
    """
    # Use a sigmoid to interpolate bounce likelihood near ground
    p_bounce = 1 / (1 + np.exp(sharpness * (z - threshold)))

    # Limit the extremes to avoid numerical issues
    p_bounce = np.clip(p_bounce, 0.01, 0.99)

    return np.array([
        [1 - p_bounce, p_bounce],
        # From flight: mostly stays in flight, some chance of bounce
        [0.1, 0.9]
        # From bounce: may return to flight (if modeling multiple bounces)
    ])

class IMMTracker:
    def __init__(self, models, dt):
        assert len(models) > 0, "There must be at least one model present."
        self.models = models
        self.P_ij = compute_transition_matrix(models[0].mu[2])
        # conditional model probabilities
        self.alpha = np.ones(len(models)) / len(models)
        self.dt = dt
        self.mu = self.models[0].mu
        self.sigma = self.models[0].sigma
        self.impact_data = None
        self.alpha_hist = []

    def set_state(self, mu, sigma):
        self.mu = mu.copy()
        self.sigma = sigma.copy()

    def get_state(self):
        return self.mu.copy(), self.sigma.copy()

    def reset(self, mu_initial, sigma_initial):
        self.mu = mu_initial
        self.sigma = sigma_initial
        self.impact_data = None

    def mix_states(self):
        N = len(self.models)
        mixed_x, mixed_P = [], []
        self.P_ij = compute_transition_matrix(self.mu[2])

        c_j = np.zeros(N)
        for j in range(N):
            c = 0
            for i in range(N):
                c += self.P_ij[i, j] * self.alpha[i]
            c_j[j] = c

        for j in range(N):
            x_j = np.zeros(6)
            P_j = np.zeros((6, 6))
            for i in range(N):
                x_i, P_i = self.models[i].get_state()
                weight = self.P_ij[i, j] * self.alpha[i] / c_j[j]
                dx = (x_i - x_j).reshape(-1, 1)
                x_j += weight * x_i
                P_j += weight * (P_i + dx @ dx.T)
            mixed_x.append(x_j)
            mixed_P.append(P_j)
        return mixed_x, mixed_P, c_j

    def predict(self):
        self.alpha_hist.append(self.alpha)
        mixed_x, mixed_P, _ = self.mix_states()
        for i, model in enumerate(self.models):
            model.set_state(mixed_x[i], mixed_P[i])
            mu, sigma = model.predict()
            model.set_state(mu, sigma)
        # the predict step in the IMM computes predictions for each model
        # hypothesis, but doesn't actually compute a prediction for the
        # overall estimator. Thus we simply return the unchanged mu and sigma
        return self.mu, self.sigma

    def update(self, cameras, y):
        # if no cameras are provided, return the unmodified mean and covariances
        if len(cameras) == 0:
            return self.mu, self.sigma

        likelihoods = []
        for i, model in enumerate(self.models):
            mu, sigma = model.update(cameras, y)
            model.set_state(mu, sigma)
            likelihoods.append(model.likelihood)
        c = np.zeros(len(self.models))
        for j in range(len(self.models)):
            c[j] = likelihoods[j] * sum(
                self.P_ij[i, j] * self.alpha[i] for i in
                range(len(self.models)))
        self.alpha = c / np.sum(c)
        mu = sum(
            self.alpha[i] * self.models[i].mu for i in range(len(self.models)))
        sigma = sum(
            self.alpha[i] * (self.models[i].sigma +
                             np.outer(self.models[i].mu - mu,
                                      self.models[i].mu - mu))
            for i in range(len(self.models))
        )
        return mu, sigma
