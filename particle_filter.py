"""
Particle Filter (Sequential Monte Carlo) for nonlinear object tracking.

Constant-velocity motion model with random perturbation to handle
nonlinear trajectories (e.g., cyclist on winding roads, drone camera motion).
Uses systematic resampling to prevent particle degeneracy.
"""

import numpy as np


class ParticleFilter:
    """2D particle filter for object tracking.

    State vector: [x, y, vx, vy]
    Particles represent hypotheses about the target's position and velocity.
    """

    def __init__(self,
                 num_particles: int = 200,
                 process_noise_pos: float = 8.0,
                 process_noise_vel: float = 3.0,
                 init_spread: float = 60.0):
        """
        Args:
            num_particles: Number of particles (hypotheses).
            process_noise_pos: Std of position noise added during prediction.
            process_noise_vel: Std of velocity noise added during prediction.
            init_spread: Std of Gaussian spread for particle initialization.
        """
        self.num_particles = num_particles
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.init_spread = init_spread

        # State: [x, y, vx, vy] for each particle
        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles
        self.initialized = False

    def init(self, x: float, y: float):
        """Initialize particles around the given position.

        Args:
            x, y: Initial target center position.
        """
        self.particles[:, 0] = x + np.random.randn(self.num_particles) * self.init_spread
        self.particles[:, 1] = y + np.random.randn(self.num_particles) * self.init_spread
        self.particles[:, 2] = np.random.randn(self.num_particles) * 3.0   # vx
        self.particles[:, 3] = np.random.randn(self.num_particles) * 3.0   # vy
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True

    def predict(self, dt: float = 1.0):
        """Propagate particles according to constant-velocity model with noise.

        The random perturbation allows the filter to handle nonlinear motion.
        """
        if not self.initialized:
            return

        # Position update:  x = x + vx*dt + noise
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 0] += np.random.randn(self.num_particles) * self.process_noise_pos

        self.particles[:, 1] += self.particles[:, 3] * dt
        self.particles[:, 1] += np.random.randn(self.num_particles) * self.process_noise_pos

        # Velocity update: perturb slightly (handles acceleration/deceleration)
        self.particles[:, 2] += np.random.randn(self.num_particles) * self.process_noise_vel
        self.particles[:, 3] += np.random.randn(self.num_particles) * self.process_noise_vel

    def update(self, observation_weights: np.ndarray):
        """Update particle weights from observation likelihoods.

        Args:
            observation_weights: Array of likelihood values for each particle.
                Should be non-negative; will be normalized internally.
        """
        if not self.initialized:
            return

        # Prevent all-zero weights
        obs = np.maximum(observation_weights, 0.0)
        if obs.sum() < 1e-10:
            obs = np.ones(self.num_particles)
        self.weights = obs / obs.sum()

    def resample(self):
        """Systematic resampling to prevent particle degeneracy.

        High-weight particles are replicated; low-weight particles are removed.
        Adds small jitter to prevent particle collapse.
        """
        if not self.initialized:
            return

        N = self.num_particles
        # Systematic resampling
        positions = (np.arange(N) + np.random.random()) / N
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Fix rounding errors

        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indexes].copy()
        self.weights = np.ones(N) / N

        # Add small jitter to maintain diversity
        self.particles[:, 0] += np.random.randn(N) * 1.5
        self.particles[:, 1] += np.random.randn(N) * 1.5
        self.particles[:, 2] += np.random.randn(N) * 0.5
        self.particles[:, 3] += np.random.randn(N) * 0.5

    def estimate(self) -> tuple[float, float]:
        """Compute weighted mean position estimate.

        Returns:
            (x, y) weighted mean of particle positions.
        """
        if not self.initialized:
            return 0.0, 0.0

        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        return float(x), float(y)

    def estimate_velocity(self) -> tuple[float, float]:
        """Compute weighted mean velocity estimate.

        Returns:
            (vx, vy) weighted mean of particle velocities.
        """
        if not self.initialized:
            return 0.0, 0.0

        vx = np.average(self.particles[:, 2], weights=self.weights)
        vy = np.average(self.particles[:, 3], weights=self.weights)
        return float(vx), float(vy)

    def get_particle_positions(self) -> np.ndarray:
        """Return all particle positions for visualization.

        Returns:
            Array of shape (num_particles, 2) with (x, y) positions.
        """
        return self.particles[:, :2].copy()

    def effective_particles(self) -> float:
        """Estimate the effective number of particles (measure of degeneracy).

        Returns:
            Effective sample size: 1 / sum(w_i^2)
        """
        if self.weights.sum() < 1e-10:
            return 0.0
        w = self.weights / self.weights.sum()
        return 1.0 / (np.sum(w ** 2) + 1e-10)
