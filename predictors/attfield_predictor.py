import numpy as np


class AttFieldPredictor:

    def __init__(self, sim_dt, obstacles, smooth_alpha=0.3, k_att=1.0, d_min=0.3):
        self._sim_dt = sim_dt
        self._alpha = smooth_alpha
        self._k_att = k_att
        self._d_min = d_min
        self._obstacles = list(obstacles)

        self._prev_pos = None
        self._vel = np.zeros(2)

    def predict(self, pos_xy, horizon_dt, N):
       
        pos = np.asarray(pos_xy, dtype=float)

        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            self._vel = self._alpha * vel_raw + (1.0 - self._alpha) * self._vel
        self._prev_pos = pos.copy()

        # v_att accumulates attraction 
        traj = np.empty((N + 1, 2))
        traj[0] = pos
        p = pos.copy()
        v_att = np.zeros(2)
        for k in range(N):
            p = p + horizon_dt * (self._vel + v_att)

            for obs in self._obstacles:
                v_att = v_att + horizon_dt * self._attractive_accel(p, obs)
                p = self._clamp_to_surface(p, obs)

            traj[k + 1] = p

        return traj

    def _directional_radii(self, u, obs):
        if hasattr(obs, 'rx'):
            
            r_surf = 1.0 / np.sqrt((u[0] / obs.rx) ** 2 + (u[1] / obs.ry) ** 2)
            rx_m = obs.rx + self._d_min
            ry_m = obs.ry + self._d_min
            r_min = 1.0 / np.sqrt((u[0] / rx_m) ** 2 + (u[1] / ry_m) ** 2)
        else:
            r_surf = obs.radius
            r_min = obs.radius + self._d_min
        return r_surf, r_min

    def _attractive_accel(self, p, obs):
        c = np.array([obs.cx, obs.cy])
        delta = c - p
        d = np.linalg.norm(delta)
        if d < 1e-6:
            return np.zeros(2)
        u = delta / d
        r_surf, r_min = self._directional_radii(u, obs)
        if d <= r_min:
            return np.zeros(2)
        surface_d = max(d - r_surf, 1e-3)
        a_mag = min(self._k_att / surface_d ** 2, 6.87)
        return a_mag * u

    def _clamp_to_surface(self, p, obs):
        c = np.array([obs.cx, obs.cy])
        delta = p - c
        d = np.linalg.norm(delta)
        if d < 1e-6:
            return p
        u = delta / d
        _, r_min = self._directional_radii(u, obs)
        if d < r_min:
            p = c + u * r_min
        return p
