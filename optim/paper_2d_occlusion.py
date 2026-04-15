import numpy as np


def _build_second_difference_matrix(N: int) -> np.ndarray:
    D2 = np.zeros((max(N - 2, 0), N))
    for i in range(N - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


class Paper2DOcclusionSolver:
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.08,
        los_samples=None,
        max_alt_iters: int = 6,
        desired_range: float = 4.0,
        min_range: float = 2.5,
        max_range: float = 5.5,
        w_smooth: float = 8.0,
        w_track: float = 120.0,
        w_occ: float = 50.0,
        w_anchor: float = 15000.0,
    ):
        self.N = int(horizon)
        self.dt = float(dt)
        self.U = np.array(
            los_samples if los_samples is not None else [0.0, 0.25, 0.5, 0.75, 1.0],
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

        self.D2 = _build_second_difference_matrix(self.N)
        self.Qs = self.D2.T @ self.D2 if self.D2.size > 0 else np.zeros((self.N, self.N))

    def _predict_target(self, target_xy, target_vel):
        tx0, ty0 = target_xy
        vx, vy = target_vel
        t = np.arange(self.N) * self.dt
        tx = tx0 + vx * t
        ty = ty0 + vy * t
        return tx, ty

    def _initial_guess(self, drone_xy, tx, ty):
        x0, y0 = drone_xy

        # # 2D adaptation of paper tracking variables:
        # initialize on the desired tracking ring around the target
        dx0 = x0 - tx[0]
        dy0 = y0 - ty[0]
        nrm = np.hypot(dx0, dy0)

        if nrm < 1e-8:
            ux0, uy0 = -1.0, 0.0
        else:
            ux0, uy0 = dx0 / nrm, dy0 / nrm

        x_ref = tx + self.desired_range * ux0
        y_ref = ty + self.desired_range * uy0

        alpha = np.linspace(0.0, 1.0, self.N)
        x = (1.0 - alpha) * x0 + alpha * x_ref
        y = (1.0 - alpha) * y0 + alpha * y_ref

        x[0] = x0
        y[0] = y0
        return x, y

    def _tracking_angle_distance_update(self, x, y, tx, ty):
        # # Paper step (27) / step (28), 2D adaptation of tracking reformulation
        # x - tx - d_r cos(alpha_r) = 0
        # y - ty - d_r sin(alpha_r) = 0
        dx = x - tx
        dy = y - ty

        alpha_r = np.arctan2(dy, dx)
        d_r = np.hypot(dx, dy)
        d_r = np.clip(d_r, self.min_range, self.max_range)

        return alpha_r, d_r

    def _occlusion_angle_distance_update(self, x, y, tx, ty, obstacles):
        # # Paper method: LOS-based occlusion formulation
        # # 2D adaptation of Eq. (13): LOS point
        # x_los = (1-u) x + u x_r
        # y_los = (1-u) y + u y_r
        #
        # # 2D adaptation of Eq. (15)-(21):
        # x_los - x_o - a d_o cos(alpha_o) = 0
        # y_los - y_o - b d_o sin(alpha_o) = 0
        # d_o >= 1
        occ_data = []

        for obs in obstacles:
            alpha_o = np.zeros((self.N, len(self.U)))
            d_o = np.ones((self.N, len(self.U)))

            for k in range(self.N):
                for j, u in enumerate(self.U):
                    x_los = (1.0 - u) * x[k] + u * tx[k]
                    y_los = (1.0 - u) * y[k] + u * ty[k]

                    rx = x_los - obs.cx
                    ry = y_los - obs.cy

                    # # Paper step (27), 2D angle update for obstacle auxiliary variable
                    alpha_o[k, j] = np.arctan2(
                        ry / max(obs.b, 1e-9),
                        rx / max(obs.a, 1e-9),
                    )

                    # # Paper step (28), 2D distance update with d_o >= 1
                    d_raw = np.sqrt(
                        (rx / max(obs.a, 1e-9)) ** 2
                        + (ry / max(obs.b, 1e-9)) ** 2
                    )
                    d_o[k, j] = max(1.0, d_raw)

            occ_data.append(
                {
                    "obs": obs,
                    "alpha_o": alpha_o,
                    "d_o": d_o,
                }
            )

        return occ_data

    def _solve_axis(self, init_val, track_rhs, occ_terms):
        # # Paper step (26): trajectory update block
        # # 2D adaptation of the paper's QP-like trajectory subproblem
        #
        # min smoothness + tracking residual + occlusion residual
        H = self.w_smooth * self.Qs.copy()
        b = np.zeros(self.N)

        H += self.w_track * np.eye(self.N)
        b += self.w_track * track_rhs

        for k_idx, coeff, rhs_scalar in occ_terms:
            H[k_idx, k_idx] += self.w_occ * (coeff ** 2)
            b[k_idx] += self.w_occ * coeff * rhs_scalar

        # current-state anchor
        H[0, 0] += self.w_anchor
        b[0] += self.w_anchor * init_val

        reg = 1e-6 * np.eye(self.N)
        sol = np.linalg.solve(H + reg, b)
        sol[0] = init_val
        return sol

    def _trajectory_update(self, drone_xy, tx, ty, alpha_r, d_r, occ_data):
        x0, y0 = drone_xy

        # # Paper method: tracking reformulation with auxiliary variables
        # 2D adaptation of Eq. (10)-(12)
        x_track_rhs = tx + d_r * np.cos(alpha_r)
        y_track_rhs = ty + d_r * np.sin(alpha_r)

        occ_terms_x = []
        occ_terms_y = []

        # # Paper method: LOS-based occlusion/collision reformulation
        # u = 0 sample also covers collision avoidance
        for item in occ_data:
            obs = item["obs"]
            alpha_o = item["alpha_o"]
            d_o = item["d_o"]

            for k in range(self.N):
                for j, u in enumerate(self.U):
                    coeff = 1.0 - u

                    rhs_x = obs.cx + obs.a * d_o[k, j] * np.cos(alpha_o[k, j]) - u * tx[k]
                    rhs_y = obs.cy + obs.b * d_o[k, j] * np.sin(alpha_o[k, j]) - u * ty[k]

                    occ_terms_x.append((k, coeff, rhs_x))
                    occ_terms_y.append((k, coeff, rhs_y))

        x_new = self._solve_axis(x0, x_track_rhs, occ_terms_x)
        y_new = self._solve_axis(y0, y_track_rhs, occ_terms_y)
        return x_new, y_new

    def solve(self, drone_xy, target_xy, target_vel, obstacles):
        tx, ty = self._predict_target(target_xy, target_vel)
        x, y = self._initial_guess(drone_xy, tx, ty)

        residual_hist = []

        # # Paper method: three-block alternating minimization
        # trajectory block -> angle block -> distance block
        for _ in range(self.max_alt_iters):
            alpha_r, d_r = self._tracking_angle_distance_update(x, y, tx, ty)
            occ_data = self._occlusion_angle_distance_update(x, y, tx, ty, obstacles)
            x, y = self._trajectory_update(drone_xy, tx, ty, alpha_r, d_r, occ_data)

            # diagnostics only
            track_res = np.mean(
                np.sqrt(
                    (x - (tx + d_r * np.cos(alpha_r))) ** 2
                    + (y - (ty + d_r * np.sin(alpha_r))) ** 2
                )
            )

            occ_res_sum = 0.0
            occ_count = 0
            for item in occ_data:
                obs = item["obs"]
                alpha_o = item["alpha_o"]
                d_o = item["d_o"]

                for k in range(self.N):
                    for j, u in enumerate(self.U):
                        x_los = (1.0 - u) * x[k] + u * tx[k]
                        y_los = (1.0 - u) * y[k] + u * ty[k]

                        rx = x_los - (obs.cx + obs.a * d_o[k, j] * np.cos(alpha_o[k, j]))
                        ry = y_los - (obs.cy + obs.b * d_o[k, j] * np.sin(alpha_o[k, j]))

                        occ_res_sum += np.hypot(rx, ry)
                        occ_count += 1

            occ_res = occ_res_sum / max(1, occ_count)
            residual_hist.append((track_res, occ_res))

        return {
            "x": x,
            "y": y,
            "tx": tx,
            "ty": ty,
            "residual_hist": residual_hist,
        }