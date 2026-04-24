# predictors/kalman_predictor.py
import numpy as np


class KalmanPredictor:
    """
    Predicts a target's future positions using a Kalman filter with a
    constant-acceleration motion model.  Maintains a 6D state
    [x, y, vx, vy, ax, ay], corrects it with each position observation,
    then rolls out the filtered state over the requested horizon.

    Parameters
    ----------
    sim_dt : float
        Time between successive calls to predict() — should match the
        simulation timestep (args.DT), not the MPC prediction timestep.
    process_noise_std : float
        Standard deviation of the jerk (rate-of-change of acceleration) noise.
        Higher values let the filter track fast manoeuvres faster at the cost
        of more noise.
    meas_noise_std : float
        Standard deviation of the position measurement noise.  Higher values
        trust observations less and smooth more aggressively.
    """

    def __init__(self, sim_dt, process_noise_std=1.0, meas_noise_std=0.5):
        self._dt = sim_dt

        # Observation model  H : (2, 6) — observe x and y only
        self._H = np.array([[1., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0.]])

        # Measurement noise covariance  R : (2, 2)
        self._R = (meas_noise_std ** 2) * np.eye(2)

        # Process noise std — stored so Q can be rebuilt per dt if needed
        self._q = process_noise_std

        # Build process noise covariance Q for sim_dt
        self._Q = self._build_Q(sim_dt, process_noise_std)

        # State mean and covariance (uninitialised until first observation)
        self._x = None          # (6,)
        self._P = None          # (6, 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pos_xy, horizon_dt, N):
        """
        Update the Kalman filter with the latest observation and return a
        predicted trajectory.

        Parameters
        ----------
        pos_xy     : array-like (2,) — current observed target position [x, y]
        horizon_dt : float — MPC prediction timestep (seconds per step)
        N          : int   — number of prediction steps

        Returns
        -------
        traj : np.ndarray, shape (N+1, 2)
            traj[0]  = current filtered position
            traj[k]  = predicted position at time k * horizon_dt from now
        """
        z = np.asarray(pos_xy, dtype=float)

        if self._x is None:
            self._initialise(z)
        else:
            self._step(z)

        # Roll out N+1 positions from current filtered state using horizon_dt
        F_h = self._build_F(horizon_dt)
        state = self._x.copy()
        traj = np.empty((N + 1, 2))
        for k in range(N + 1):
            traj[k] = state[:2]
            state = F_h @ state
        return traj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialise(self, z):
        """Seed the filter from the first observation (zero velocity and acceleration)."""
        self._x = np.array([z[0], z[1], 0., 0., 0., 0.])
        # Large initial position uncertainty; moderate velocity/acceleration uncertainty
        self._P = np.diag([1e2, 1e2, 1e1, 1e1, 1e1, 1e1])

    def _step(self, z):
        """One predict-update cycle of the Kalman filter at sim_dt."""
        F = self._build_F(self._dt)

        # --- Predict ---
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + self._Q

        # --- Update ---
        S = self._H @ P_pred @ self._H.T + self._R          # (2, 2)
        K = P_pred @ self._H.T @ np.linalg.inv(S)           # (6, 2)
        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (np.eye(6) - K @ self._H) @ P_pred

    @staticmethod
    def _build_F(dt):
        """Constant-acceleration state transition matrix for timestep dt."""
        dt2 = 0.5 * dt * dt
        return np.array([
            [1., 0., dt, 0., dt2, 0.],
            [0., 1., 0., dt, 0., dt2],
            [0., 0., 1., 0., dt,  0.],
            [0., 0., 0., 1., 0.,  dt],
            [0., 0., 0., 0., 1.,  0.],
            [0., 0., 0., 0., 0.,  1.],
        ])

    @staticmethod
    def _build_Q(dt, q_std):
        """
        Discrete-time process noise covariance for a constant-acceleration model
        driven by white jerk noise with standard deviation q_std.
        """
        # Continuous-time jerk noise integrated over dt for one axis:
        # state = [p, v, a], noise input on da/dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        q2 = q_std ** 2
        # Single-axis block (p, v, a) process noise covariance
        block = q2 * np.array([
            [dt5 / 20., dt4 / 8., dt3 / 6.],
            [dt4 / 8.,  dt3 / 3., dt2 / 2.],
            [dt3 / 6.,  dt2 / 2., dt],
        ])
        Q = np.zeros((6, 6))
        # x-axis: indices 0 (p), 2 (v), 4 (a)
        ix = np.array([0, 2, 4])
        Q[np.ix_(ix, ix)] = block
        # y-axis: indices 1 (p), 3 (v), 5 (a)
        iy = np.array([1, 3, 5])
        Q[np.ix_(iy, iy)] = block
        return Q
