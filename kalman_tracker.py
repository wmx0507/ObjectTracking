"""
Kalman filter for object position prediction.
Constant velocity model: state = [x, y, vx, vy]
"""

import numpy as np


class KalmanTracker:
    """2D Kalman filter with constant velocity model."""

    def __init__(self, dt: float = 1.0,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1):
        # State: [x, y, vx, vy]
        self.state = np.zeros((4, 1))
        self.covariance = np.eye(4) * 100.0

        self.dt = dt

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2]
        ])

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        self.initialized = False

    def init(self, x: float, y: float):
        """Initialize state with first measurement."""
        self.state[0, 0] = x
        self.state[1, 0] = y
        self.state[2, 0] = 0.0
        self.state[3, 0] = 0.0
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        """Predict next state. Returns (x, y)."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[0, 0], self.state[1, 0]

    def update(self, x: float, y: float):
        """Update state with measurement."""
        z = np.array([[x], [y]])
        y_err = z - self.H @ self.state  # innovation
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y_err
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
