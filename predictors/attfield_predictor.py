import numpy as np


class AttFieldPredictor:
    """
    Predicts a target's future positions using a constant-velocity base rollout
    augmented by per-obstacle attractive potential fields.  At each prediction
    step the rollout position is pulled toward every obstacle center; a
    minimum-distance clamp then ensures no predicted position lands inside or
    too close to an obstacle.

    Parameters
    ----------
    sim_dt : float
        Time between successive calls to predict() — should match the
        simulation timestep (args.DT), not the MPC prediction timestep.
    obstacles : list
        List of CircleObstacle / EllipseObstacle instances from base_env.py.
        Each must expose (cx, cy) and either .radius (circle) or .rx/.ry
        (ellipse).
    smooth_alpha : float
        EMA weight on the newest velocity sample (0 = frozen, 1 = raw FD).
    k_att : float
        Attractive gain.  Displacement per step saturates at k_att * horizon_dt
        (metres) as distance grows large; near the surface it approaches zero.
    d_min : float
        Minimum clearance (metres) from the obstacle surface.  Predicted
        positions are projected outward so they are never closer than this.
    """

    def __init__(self, sim_dt, obstacles, smooth_alpha=0.3, k_att=1.0, d_min=0.3):
        self._sim_dt = sim_dt
        self._alpha = smooth_alpha
        self._k_att = k_att
        self._d_min = d_min
        self._obstacles = list(obstacles)

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

        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            self._vel = self._alpha * vel_raw + (1.0 - self._alpha) * self._vel
        self._prev_pos = pos.copy()

        # v_att accumulates attraction like gravity: each step adds acceleration
        # to the running velocity so the curve bends progressively, not abruptly.
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
        """
        Return (r_surf, r_min) along unit direction u away from obstacle center.

        For circles these are constant.  For ellipses, r_surf is the distance
        from the center to the ellipse boundary in direction u, and r_min is
        the same quantity for the expanded ellipse (rx+d_min, ry+d_min).
        """
        if hasattr(obs, 'rx'):
            # Elliptical boundary in direction u:
            #   solve t s.t. (t*u[0]/rx)^2 + (t*u[1]/ry)^2 = 1  =>  t = r_surf
            r_surf = 1.0 / np.sqrt((u[0] / obs.rx) ** 2 + (u[1] / obs.ry) ** 2)
            rx_m = obs.rx + self._d_min
            ry_m = obs.ry + self._d_min
            r_min = 1.0 / np.sqrt((u[0] / rx_m) ** 2 + (u[1] / ry_m) ** 2)
        else:
            r_surf = obs.radius
            r_min = obs.radius + self._d_min
        return r_surf, r_min

    def _attractive_accel(self, p, obs):
        """Return acceleration vector (m/s²) pulling p toward obs."""
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
        """Project p outward if it has landed inside the minimum safe boundary."""
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
