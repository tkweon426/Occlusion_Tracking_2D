import numpy as np


class VelAccPredictor:

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
