import numpy as np


class VelAccPredictor:
    """
    Predicts a target's future positions by estimating velocity and acceleration
    via exponentially-smoothed finite differences, then rolling out with constant
    acceleration over the requested horizon.

    Parameters
    ----------
    sim_dt : float
        Time between successive calls to predict() — should match the
        simulation timestep (args.DT), not the MPC prediction timestep.
    vel_alpha : float
        EMA weight on the newest velocity sample (0 = frozen, 1 = raw FD).
    acc_alpha : float
        EMA weight on the newest acceleration sample (0 = frozen, 1 = raw FD).
    warmup_steps : int
        Number of velocity samples to collect before trusting the acceleration
        estimate. During warmup the rollout falls back to constant velocity,
        preventing the large spike caused by velocity jumping from/to zero at
        the start and end of a trajectory.
    acc_clip : float or None
        If set, raw finite-difference acceleration is clipped to this magnitude
        (m/s²) before being fed into the EMA. Guards against single-step
        outliers (e.g. abrupt stop) overwhelming the smoothed estimate.
    """

    def __init__(self, sim_dt, vel_alpha=0.3, acc_alpha=0.2, warmup_steps=5, acc_clip=10.0):
        self._sim_dt = sim_dt
        self._vel_alpha = vel_alpha
        self._acc_alpha = acc_alpha
        self._warmup_steps = warmup_steps
        self._acc_clip = acc_clip

        self._prev_pos = None
        self._vel = np.zeros(2)
        self._prev_vel = None
        self._acc = np.zeros(2)
        self._vel_sample_count = 0

    def predict(self, pos_xy, horizon_dt, N):
        """
        Update the velocity and acceleration estimates and return a predicted trajectory.

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

        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            new_vel = self._vel_alpha * vel_raw + (1.0 - self._vel_alpha) * self._vel
            self._vel_sample_count += 1

            if self._prev_vel is not None and self._vel_sample_count >= self._warmup_steps:
                acc_raw = (new_vel - self._prev_vel) / self._sim_dt
                if self._acc_clip is not None:
                    mag = np.linalg.norm(acc_raw)
                    if mag > self._acc_clip:
                        acc_raw = acc_raw * (self._acc_clip / mag)
                self._acc = self._acc_alpha * acc_raw + (1.0 - self._acc_alpha) * self._acc

            self._prev_vel = self._vel.copy()
            self._vel = new_vel

        self._prev_pos = pos.copy()

        steps = np.arange(N + 1, dtype=float)
        t = steps * horizon_dt
        traj = (
            pos
            + t[:, None] * self._vel
            + 0.5 * (t**2)[:, None] * self._acc
        )
        return traj
