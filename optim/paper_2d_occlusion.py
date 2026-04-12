import numpy as np


def _build_second_difference_matrix(N):
    if N < 3:
        return np.zeros((0, N))
    D2 = np.zeros((N - 2, N))
    for i in range(N - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


class Paper2DOcclusionSolver:
    def __init__(
        self,
        horizon=30,
        dt=0.08,
        los_samples=None,
        max_alt_iters=8,
        desired_range=4.5,
        min_range=3.0,
        max_range=6.0,
        w_smooth=12.0,
        w_track=120.0,
        w_occ=25.0,
        w_anchor=5000.0,
        w_prox=0.0,
    ):
        self.N = int(horizon)
        self.dt = float(dt)
        self.U = np.array(
            los_samples if los_samples is not None else [0.0, 0.15, 0.35, 0.55, 0.75, 1.0],
            dtype=float,
        )
        self.max_alt_iters = int(max_alt_iters)

        self.desired_range = float(desired_range)
        self.min_range = float(min_range)
        self.max_range = float(max_range)

        self.w_smooth = float(w_smooth)
        self.w_track = float(w_track)
        self.w_occ = float(w_occ)
        self.w_anchor = float(w_anchor)
        self.w_prox = float(w_prox)

        self.D2 = _build_second_difference_matrix(self.N)
        self.Qs = self.D2.T @ self.D2 if self.D2.size > 0 else np.zeros((self.N, self.N))

        self.prev_x = None
        self.prev_y = None

    def _predict_target(self, target_xy, target_vel):
        tx0, ty0 = target_xy
        vx, vy = target_vel
        t = np.arange(self.N) * self.dt
        tx = tx0 + vx * t
        ty = ty0 + vy * t
        return tx, ty

    def _init_traj(self, drone_xy, tx, ty):
        x0, y0 = drone_xy

        dx = x0 - tx[0]
        dy = y0 - ty[0]
        nrm = np.hypot(dx, dy)

        if nrm < 1e-6:
            dx, dy = -1.0, 0.0
            nrm = 1.0

        ux = dx / nrm
        uy = dy / nrm

        x_des = tx + self.desired_range * ux
        y_des = ty + self.desired_range * uy

        #if self.prev_x is not None and self.prev_y is not None and len(self.prev_x) == self.N:
            #return self.prev_x.copy(), self.prev_y.copy()

        alpha = np.linspace(0.0, 1.0, self.N)
        x = (1.0 - alpha) * x0 + alpha * x_des
        y = (1.0 - alpha) * y0 + alpha * y_des
        x[0] = x0
        y[0] = y0
        return x, y

    def _update_tracking_angles_and_dist(self, x, y, tx, ty):
        dx = x - tx
        dy = y - ty
        alpha_r = np.arctan2(dy, dx)

        d_raw = np.hypot(dx, dy)
        d_r = np.clip(d_raw, self.min_range, self.max_range)
        return alpha_r, d_r

    def _update_occ_angles_and_dist(self, x, y, tx, ty, obstacles):
        occ_data = []

        for obs in obstacles:
            alpha_o = np.zeros((self.N, len(self.U)))
            d_o = np.ones((self.N, len(self.U)))

            for k in range(self.N):
                for j, u in enumerate(self.U):
                    xlos = (1.0 - u) * x[k] + u * tx[k]
                    ylos = (1.0 - u) * y[k] + u * ty[k]

                    xr = xlos - obs.cx
                    yr = ylos - obs.cy

                    alpha_o[k, j] = np.arctan2(yr / obs.b, xr / obs.a)
                    d_raw = np.sqrt((xr / obs.a) ** 2 + (yr / obs.b) ** 2)
                    d_o[k, j] = max(1.0, d_raw)

            occ_data.append(
                {
                    "obs": obs,
                    "alpha_o": alpha_o,
                    "d_o": d_o,
                }
            )

        return occ_data

    def _solve_axis(self, init_val, target_rhs, occ_terms, prev_axis):
        H = self.w_smooth * self.Qs.copy()
        b = np.zeros(self.N)

        H += self.w_track * np.eye(self.N)
        b += self.w_track * target_rhs

        for coeff_vec, rhs_vec in occ_terms:
            H += self.w_occ * np.diag(coeff_vec ** 2)
            b += self.w_occ * (coeff_vec * rhs_vec)

        H[0, 0] += self.w_anchor
        b[0] += self.w_anchor * init_val

        if prev_axis is not None:
            H += self.w_prox * np.eye(self.N)
            b += self.w_prox * prev_axis

        reg = 1e-6 * np.eye(self.N)
        sol = np.linalg.solve(H + reg, b)
        sol[0] = init_val
        # keep trajectory inside a reasonable workspace
        sol = np.clip(sol, -12.0, 12.0)
        # keep first point exactly at current state
        sol[0] = init_val
        
        return sol

    def _trajectory_update(self, drone_xy, tx, ty, alpha_r, d_r, occ_data, x_prev, y_prev):
        x0, y0 = drone_xy

        x_track_rhs = tx + d_r * np.cos(alpha_r)
        y_track_rhs = ty + d_r * np.sin(alpha_r)

        occ_terms_x = []
        occ_terms_y = []

        for item in occ_data:
            obs = item["obs"]
            alpha_o = item["alpha_o"]
            d_o = item["d_o"]

            for k in range(self.N):
                for j, u in enumerate(self.U):
                    c = 1.0 - u

                    rhs_x = obs.cx + obs.a * d_o[k, j] * np.cos(alpha_o[k, j]) - u * tx[k]
                    rhs_y = obs.cy + obs.b * d_o[k, j] * np.sin(alpha_o[k, j]) - u * ty[k]

                    coeff_x = np.zeros(self.N)
                    coeff_y = np.zeros(self.N)
                    coeff_x[k] = c
                    coeff_y[k] = c

                    rhs_vec_x = np.zeros(self.N)
                    rhs_vec_y = np.zeros(self.N)
                    rhs_vec_x[k] = rhs_x
                    rhs_vec_y[k] = rhs_y

                    occ_terms_x.append((coeff_x, rhs_vec_x))
                    occ_terms_y.append((coeff_y, rhs_vec_y))

        x_new = self._solve_axis(x0, x_track_rhs, occ_terms_x, x_prev)
        y_new = self._solve_axis(y0, y_track_rhs, occ_terms_y, y_prev)
        return x_new, y_new

    def solve(self, drone_xy, target_xy, target_vel, obstacles):
        tx, ty = self._predict_target(target_xy, target_vel)
        x, y = self._init_traj(drone_xy, tx, ty)

        residual_hist = []

        for _ in range(self.max_alt_iters):
            alpha_r, d_r = self._update_tracking_angles_and_dist(x, y, tx, ty)
            occ_data = self._update_occ_angles_and_dist(x, y, tx, ty, obstacles)
            x, y = self._trajectory_update(drone_xy, tx, ty, alpha_r, d_r, occ_data, x, y)

            track_res = np.mean(
                np.sqrt((x - (tx + d_r * np.cos(alpha_r))) ** 2 + (y - (ty + d_r * np.sin(alpha_r))) ** 2)
            )

            occ_res_sum = 0.0
            occ_count = 0
            for item in occ_data:
                obs = item["obs"]
                alpha_o = item["alpha_o"]
                d_o = item["d_o"]
                for k in range(self.N):
                    for j, u in enumerate(self.U):
                        xlos = (1.0 - u) * x[k] + u * tx[k]
                        ylos = (1.0 - u) * y[k] + u * ty[k]
                        rx = xlos - (obs.cx + obs.a * d_o[k, j] * np.cos(alpha_o[k, j]))
                        ry = ylos - (obs.cy + obs.b * d_o[k, j] * np.sin(alpha_o[k, j]))
                        occ_res_sum += np.hypot(rx, ry)
                        occ_count += 1

            occ_res = occ_res_sum / max(1, occ_count)
            residual_hist.append((track_res, occ_res))

        self.prev_x = x.copy()
        self.prev_y = y.copy()

        return {
            "x": x,
            "y": y,
            "tx": tx,
            "ty": ty,
            "residual_hist": residual_hist,
        }