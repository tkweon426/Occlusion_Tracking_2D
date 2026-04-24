# predictors/attfield_predictor.py
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
        Attractive gain.  The maximum displacement applied per obstacle per
        prediction step is k_att * horizon_dt (metres), reached when the
        rollout point is exactly on the obstacle surface.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # --- Velocity estimation (EMA finite difference) ---
        if self._prev_pos is not None:
            vel_raw = (pos - self._prev_pos) / self._sim_dt
            self._vel = self._alpha * vel_raw + (1.0 - self._alpha) * self._vel
        self._prev_pos = pos.copy()

        # --- Iterative rollout with attractive field ---
        traj = np.empty((N + 1, 2))
        traj[0] = pos
        p = pos.copy()
        for _ in range(N):
            # Base constant-velocity step
            p = p + horizon_dt * self._vel

            # Apply attractive displacement and surface clamp per obstacle
            for obs in self._obstacles:
                p = self._apply_obstacle(p, obs, horizon_dt)

            traj[_ + 1] = p

        return traj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _apply_obstacle(self, p, obs, horizon_dt):
        """Apply attraction toward obs and clamp to minimum safe distance."""
        c = np.array([obs.cx, obs.cy])
        delta = c - p          # vector from p toward center
        d = np.linalg.norm(delta)

        if d < 1e-6:
            return p

        u = delta / d          # unit vector toward center
        r_surf, r_min = self._directional_radii(u, obs)
        surface_d = d - r_surf

        # Attract when outside the minimum safe zone
        if d > r_min:
            # Bounded magnitude: max = k_att * horizon_dt near the surface
            F_mag = self._k_att / (1.0 + surface_d ** 2)
            p = p + F_mag * u * horizon_dt

        # Clamp: project outward if inside minimum safe boundary
        delta_new = p - c
        d_new = np.linalg.norm(delta_new)
        if d_new > 1e-6:
            u_new = delta_new / d_new
            _, r_min_new = self._directional_radii(u_new, obs)
            if d_new < r_min_new:
                p = c + u_new * r_min_new

        return p
