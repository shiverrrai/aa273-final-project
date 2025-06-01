import numpy as np
import constants as consts


def bundle_visible_measurements(cameras, y, visibility):
    """
    Bundles the visible (non-Nan) measurements received at a given time step.
    :param cameras: list of PinholeCamera objects (length m)
    :param y: (mx2) ndarray of measurements
    :param visibility: (mx1) ndarray of bools representing whether
    measurements are visible
    :return: valid_cameras, visible_measurements
    """
    valid_cameras = []
    visible_measurements = []
    for i in range(len(cameras)):
        if visibility[i]:
            visible_measurements.append(y[i, :])
            valid_cameras.append(cameras[i])
    return valid_cameras, np.asarray(visible_measurements)


def check_ground_impact(prev_particles, prev_weights, curr_particles, curr_weights, dt):
    """
    Checks whether the particle cloud has made impact with the ground.
    Uses weighted average of particles to estimate impact location.

    :param prev_particles: previous particle cloud, (6, n_particles) array
    :param prev_weights: previous particle weights, (n_particles,) array  
    :param curr_particles: current particle cloud, (6, n_particles) array
    :param curr_weights: current particle weights, (n_particles,) array
    :param dt: time step
    :return: np array of x & y location, along with covariance matrix at
    moment of impact
    """
    # Get weighted mean states
    prev_state = np.average(prev_particles, axis=1, weights=prev_weights)
    curr_state = np.average(curr_particles, axis=1, weights=curr_weights)
    
    z0, z1 = prev_state[2], curr_state[2]
    x0, x1 = prev_state[0], curr_state[0]

    # Search for bounce in a reasonable position window
    buffer = 1
    position_window = [consts.court_length / 2, consts.court_length + buffer]
    if position_window[0] <= x1 <= position_window[1] and (np.sign(z0) <= 0 or np.sign(z1) <= 0):
        alpha = z0 / (z0 - z1)
        
        # Interpolate particle positions at impact
        impact_particles = prev_particles + alpha * (curr_particles - prev_particles)
        impact_weights = (prev_weights + curr_weights) / 2  # Average weights
        
        # Get impact location and uncertainty from particles
        x_impact = np.average(impact_particles[0], weights=impact_weights)
        y_impact = np.average(impact_particles[1], weights=impact_weights)
        
        # Compute covariance from particles
        impact_xy = impact_particles[:2, :]  # Just x,y positions
        sigma_xy = np.cov(impact_xy, aweights=impact_weights)

        return np.array([x_impact, y_impact, sigma_xy], dtype=object)
    return None


class ParticleFilter:
    """
    Particle Filter for tennis ball tracking
    """

    def __init__(self, mu_initial, sigma_initial, Q, R, dt, n_particles=5000):
        """
        Initialize the Particle Filter

        :param mu_initial: initial guess for state vector (6,)
        :param sigma_initial: initial guess for covariance matrix (6,6)
        :param Q: process noise covariance matrix (6,6)
        :param R: measurement noise covariance matrix (4,4 for 2 cameras)
        :param dt: time step
        :param n_particles: number of particles
        """
        self.mu = np.array(mu_initial)
        self.sigma = np.array(sigma_initial)
        self.Q = Q
        self.R = R
        self.dt = dt
        self.n_particles = n_particles
        self.impact_data = None
        
        # Initialize particles and weights
        self.particles = None
        self.weights = None
        self.initialize_particles()
        
        self.eps = 2 * np.finfo(float).eps

    def reset(self, mu_initial, sigma_initial):
        """Reset filter to initial conditions"""
        self.mu = np.array(mu_initial)
        self.sigma = np.array(sigma_initial)
        self.impact_data = None
        self.initialize_particles()

    def initialize_particles(self):
        """Initialize particle cloud from initial distribution"""
        self.particles = np.random.multivariate_normal(
            self.mu, self.sigma, size=self.n_particles
        ).T  # Shape: (6, n_particles)
        
        self.weights = np.ones(self.n_particles) / self.n_particles
        assert self.particles.shape == (6, self.n_particles)
        assert np.isclose(self.weights.sum(), 1), "Weights do not sum to 1"

    def f(self, particles):
        """
        Propagate particles through dynamics (vectorized)
        
        :param particles: (6, n_particles) array
        :return: propagated particles (6, n_particles)
        """
        new_particles = particles.copy()
        
        # Update positions: x += dt * xdot
        new_particles[0] += self.dt * particles[3]  # x
        new_particles[1] += self.dt * particles[4]  # y  
        new_particles[2] += self.dt * particles[5] + 0.5 * consts.g * (self.dt ** 2)  # z
        
        # Update velocities (only z-velocity changes due to gravity)
        new_particles[5] += consts.g * self.dt  # zdot
        
        return new_particles

    def sample_f(self, particles):
        """
        Sample from stochastic dynamics (adds process noise)
        
        :param particles: (6, n_particles) array
        :return: noisy propagated particles (6, n_particles)
        """
        # Deterministic propagation
        propagated = self.f(particles)
        
        # Add process noise
        noise = np.random.multivariate_normal(
            np.zeros(6), self.Q, size=self.n_particles
        ).T  # Shape: (6, n_particles)
        
        return propagated + noise

    def measurement_likelihood(self, y, cameras, particles):
        """
        Compute measurement likelihood for each particle
        
        :param y: measurements from visible cameras (n_cameras, 2)
        :param cameras: list of visible camera objects
        :param particles: (6, n_particles) array
        :return: likelihood for each particle (n_particles,)
        """
        n_particles = particles.shape[1]
        n_cameras = len(cameras)
        
        if n_cameras == 0:
            return np.ones(n_particles)  # No measurements, all particles equally likely
        
        # Get measurement noise variance
        measurement_noise_var = self.R[0, 0]  # Assuming diagonal R matrix
        
        # Use log-likelihoods for numerical stability
        log_likelihoods = np.zeros(n_particles)
        
        for i, cam in enumerate(cameras):
            measurement = y[i]  # (2,) pixel measurement
            
            for p in range(n_particles):
                particle_pos = particles[:3, p]  # Extract position
                pixel, is_visible = cam.g(particle_pos)
                
                if is_visible:
                    # Compute squared error
                    error = pixel - measurement
                    squared_error = np.sum(error**2)
                    
                    # Add log-likelihood (Gaussian)
                    log_likelihoods[p] += -squared_error / (2 * measurement_noise_var)
                else:
                    # If particle not visible, give very low likelihood
                    log_likelihoods[p] += -1000  # Very negative log-likelihood
        
        # Convert back to likelihoods with numerical stability
        # Subtract max log-likelihood to prevent overflow
        max_log_likelihood = np.max(log_likelihoods)
        log_likelihoods -= max_log_likelihood
        
        likelihoods = np.exp(log_likelihoods) + self.eps
        
        return likelihoods

    def predict(self):
        """PF Predict Step"""
        self.particles = self.sample_f(self.particles)
        return self.particles

    def update(self, cameras, y):
        """PF Update Step"""
        if len(cameras) == 0:
            return self.weights
        
        # Compute likelihoods
        likelihoods = self.measurement_likelihood(y, cameras, self.particles)
        
        # Update weights
        self.weights = self.weights * likelihoods
        
        # Handle weight underflow/overflow
        weight_sum = self.weights.sum()
        if weight_sum < self.eps:
            # All weights underflowed - reset to uniform
            print("Warning: All particle weights underflowed, resetting to uniform")
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            # Normalize weights
            self.weights = self.weights / weight_sum
        
        assert np.isclose(self.weights.sum(), 1), f"Weights sum to {self.weights.sum()}, not 1"
        return self.weights

    def resample(self):
        """PF Resample Step using importance resampling"""
        # Check if resampling is needed (effective sample size)
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        
        if effective_sample_size < self.n_particles / 2:
            # Resample particles
            indices = np.random.choice(
                np.arange(self.n_particles),
                size=self.n_particles,
                p=self.weights
            )
            
            self.particles = self.particles[:, indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        return self.particles, self.weights

    def get_curr_mu_sigma(self):
        """Get current weighted mean and covariance"""
        mu = np.average(self.particles, axis=1, weights=self.weights)
        sigma = np.cov(self.particles, aweights=self.weights)
        return mu, sigma

    def run(self, cameras, y, visibility):
        """
        Run particle filter on measurement sequence
        
        :param cameras: list of PinholeCamera objects
        :param y: (m, n, 2) measurement array
        :param visibility: (m, n) visibility array
        :return: (n, 6) mean estimates, (n, 6, 6) covariances
        """
        mu_hist = []
        sigma_hist = []
        
        for i in range(y.shape[1]):
            # Store previous state for impact detection
            prev_particles = self.particles.copy()
            prev_weights = self.weights.copy()
            
            # PF Steps
            self.predict()
            
            # Bundle visible measurements
            valid_cameras, visible_measurements = bundle_visible_measurements(
                cameras, y[:, i, :], visibility[:, i]
            )
            
            # # Debug info
            # if i < 5 or i % 50 == 0:  # Print debug for first few timesteps and every 50th
            #     print(f"PF timestep {i}: {len(valid_cameras)} visible cameras")
            #     if len(valid_cameras) > 0:
            #         print(f"  Measurements: {visible_measurements}")
            #         print(f"  Particle mean position: {np.mean(self.particles[:3], axis=1)}")
            
            self.update(valid_cameras, visible_measurements)
            self.resample()
            
            # Update state estimates
            self.mu, self.sigma = self.get_curr_mu_sigma()
            
            # Check for ground impact
            if i > 0:  # Need previous state
                impact = check_ground_impact(
                    prev_particles, prev_weights,
                    self.particles, self.weights, self.dt
                )
                if impact is not None and self.impact_data is None:
                    self.impact_data = impact
            
            # Store history
            mu_hist.append(self.mu.copy())
            sigma_hist.append(self.sigma.copy())

        return np.array(mu_hist), np.array(sigma_hist)