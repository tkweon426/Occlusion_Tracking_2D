# predictors/constvel_predictor.py
import numpy as np


class ConstVelPredictor:
    """
    Predicts a target's future positions by estimating its velocity via
    exponentially-smoothed finite differences between successive observations,
    then rolling out at constant velocity over the requested horizon.

    Parameters
    ----------
    sim_dt : float
        Time between successive calls to predict() — should match the
        simulation timestep (args.DT), not the MPC prediction timestep.
    smooth_alpha : float
        EMA weight on the newest velocity sample (0 = frozen, 1 = raw FD).
        Lower values filter noise; higher values track fast manoeuvres sooner.
    """

    def __init__(self, sim_dt, smooth_alpha=0.3):
        self._sim_dt = sim_dt
        self._alpha = smooth_alpha
        self._prev_pos = None
        self._vel = np.zeros(2)

    def predict(self, pos_xy, horizon_dt, N):
        """
        Update the velocity estimate and return a predicted trajectory.

        Parameters
        ----------
        pos_xy     : array-like (2,) — current observed target position [x, y]
        horizon_dt : float — MPC prediction timestep (seconds per step)
        N          : int   — number of prediction steps

        Returns
        -------
        traj : np.ndarray, shape (N+1, 2)
            traj[0]  = current position (observed)
            traj[k]  = predicted position at time k * horizon_dt from now
        """
        pos = np.asarray(pos_xy, dtype=float)

        # Update smoothed velocity estimate via finite difference
        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            self._vel = self._alpha * vel_raw + (1.0 - self._alpha) * self._vel
        self._prev_pos = pos.copy()

        # Constant-velocity rollout: traj[k] = pos + k * horizon_dt * vel
        steps = np.arange(N + 1, dtype=float)          # shape (N+1,)
        traj = pos + steps[:, None] * horizon_dt * self._vel   # shape (N+1, 2)
        return traj
