import numpy as np


class ConstVelPredictor:

    def __init__(self, sim_dt, smooth_alpha=0.3):
        self._sim_dt = sim_dt
        self._alpha = smooth_alpha
        self._prev_pos = None
        self._vel = np.zeros(2)

    def predict(self, pos_xy, horizon_dt, N):

        pos = np.asarray(pos_xy, dtype=float)

        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            self._vel = self._alpha * vel_raw + (1.0 - self._alpha) * self._vel
        self._prev_pos = pos.copy()

        steps = np.arange(N + 1, dtype=float)
        traj = pos + steps[:, None] * horizon_dt * self._vel
        return traj
