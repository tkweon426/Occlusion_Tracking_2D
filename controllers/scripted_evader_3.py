import numpy as np


class ScriptedTrajectory_3:
    _REACH_THRESHOLD = 0.25

    def __init__(
        self,
        obstacle_cx: float = 3.0,
        obstacle_cy: float = 7.0,
        speed: float = 2.0,
        clearance: float = 0.5,
        circle_radius: float = 1.0,
        ellipse_cx: float = -4.0,
        ellipse_cy: float = 0.0,
        ellipse_rx: float = 1.0,
        ellipse_ry: float = 3.0,
        n_line: int = 40,
        n_circle: int = 120,
        n_ellipse: int = 160,
    ):
        self.speed = speed
        self.clearance = clearance

        self.circle_cx = obstacle_cx
        self.circle_cy = obstacle_cy
        self.circle_radius = circle_radius

        self.ellipse_cx = ellipse_cx
        self.ellipse_cy = ellipse_cy
        self.ellipse_rx = ellipse_rx
        self.ellipse_ry = ellipse_ry

        self.circle_path_radius = circle_radius + clearance
        self.ellipse_path_rx = ellipse_rx + clearance
        self.ellipse_path_ry = ellipse_ry + clearance

        self.n_line = n_line
        self.n_circle = n_circle
        self.n_ellipse = n_ellipse

        self._waypoints = self._build_waypoints()
        self._idx = 0

    def _add_line(self, wps, start, end, speed=None, n=None):
        if speed is None:
            speed = self.speed
        if n is None:
            n = self.n_line

        sx, sy = start
        ex, ey = end

        for i in range(1, n + 1):
            a = i / n
            x = sx + a * (ex - sx)
            y = sy + a * (ey - sy)
            wps.append((x, y, speed))

    def _add_circle_arc(self, wps, theta0, theta1, speed=None, n=None):
        if speed is None:
            speed = self.speed
        if n is None:
            n = self.n_circle

        cx = self.circle_cx
        cy = self.circle_cy
        r = self.circle_path_radius

        for i in range(1, n + 1):
            a = i / n
            theta = theta0 + a * (theta1 - theta0)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            wps.append((x, y, speed))

    def _add_ellipse_arc(self, wps, theta0, theta1, speed=None, n=None):
        if speed is None:
            speed = self.speed
        if n is None:
            n = self.n_ellipse

        cx = self.ellipse_cx
        cy = self.ellipse_cy
        rx = self.ellipse_path_rx
        ry = self.ellipse_path_ry

        for i in range(1, n + 1):
            a = i / n
            theta = theta0 + a * (theta1 - theta0)
            x = cx + rx * np.cos(theta)
            y = cy + ry * np.sin(theta)
            wps.append((x, y, speed))

    def _build_waypoints(self):
        wps = []

        current = (0.0, 3.0)

        circle_entry_angle = -3.0 * np.pi / 4.0

        circle_entry = (
            self.circle_cx + self.circle_path_radius * np.cos(circle_entry_angle),
            self.circle_cy + self.circle_path_radius * np.sin(circle_entry_angle),
        )

        self._add_line(wps, current, circle_entry, n=45)
        current = circle_entry

        self._add_circle_arc(
            wps,
            theta0=circle_entry_angle,
            theta1=circle_entry_angle - 2.0 * np.pi,
            n=self.n_circle,
        )

        current = circle_entry

        ellipse_top = (
            self.ellipse_cx,
            self.ellipse_cy + self.ellipse_path_ry,
        )

        self._add_line(wps, current, ellipse_top, n=65)
        current = ellipse_top

        self._add_ellipse_arc(
            wps,
            theta0=np.pi / 2.0,
            theta1=np.pi / 2.0 + 2.0 * np.pi,
            n=self.n_ellipse,
        )

        current = ellipse_top

        exit_1 = (-2.0, 4.0)
        exit_2 = (4.0, 3.5)
        exit_3 = (9.0, 4.0)

        self._add_line(wps, current, exit_1, n=35)
        self._add_line(wps, exit_1, exit_2, n=45)
        self._add_line(wps, exit_2, exit_3, n=35)

        return wps

    @property
    def done(self) -> bool:
        return self._idx >= len(self._waypoints)

    def reset(self):
        self._idx = 0

    def get_velocity(self, state):
        if self.done:
            return 0.0, 0.0

        x = float(state[0])
        y = float(state[1])

        wx, wy, speed = self._waypoints[self._idx]

        dx = wx - x
        dy = wy - y
        dist = np.hypot(dx, dy)

        while dist < self._REACH_THRESHOLD:
            self._idx += 1

            if self.done:
                return 0.0, 0.0

            wx, wy, speed = self._waypoints[self._idx]
            dx = wx - x
            dy = wy - y
            dist = np.hypot(dx, dy)

        if dist < 1e-9:
            return 0.0, 0.0

        vx = speed * dx / dist
        vy = speed * dy / dist

        return float(vx), float(vy)