# controllers/scripted_evader.py
# Pre-planned trajectory for the evader: approach → clockwise orbit → straight line → decelerate to stop.

import numpy as np

class ScriptedTrajectory_2:
    """
    A time-indexed waypoint trajectory. The evader is steered toward successive
    waypoints; each waypoint carries a target speed so you get smooth deceleration.

    Phases
    ------
    1. Approach   - move from spawn toward (0, 7), the orbit entry point.
    2. Orbit      - one full clockwise loop around the obstacle at (3, 7).
    3. Straight   - travel in a straight line away from the obstacle.
    4. Decelerate - slow to a stop at the destination.
    """

    # How close (metres) the evader must be before it advances to the next waypoint.
    _REACH_THRESHOLD = 0.25

    def __init__(
        self,
        obstacle_cx: float = 3.0,
        obstacle_cy: float = 7.0,
        orbit_radius: float = 3.0,    # distance from obstacle centre to orbit path
        orbit_speed: float = 2.68,    # m/s while looping
        travel_speed: float = 2.68,   # m/s on the straight-line leg
        stop_point: tuple = (-8.0, -4.0),
        orbit_points: int = 72,      # resolution of the circular arc (one per 5 deg)
    ):
        self._cx = obstacle_cx
        self._cy = obstacle_cy
        self._r = orbit_radius
        self._orbit_speed = orbit_speed
        self._travel_speed = travel_speed
        self._stop = stop_point
        self._n = orbit_points

        self._waypoints = self._build_waypoints()
        self._idx = 0

    # ------------------------------------------------------------------
    # Waypoint construction
    # ------------------------------------------------------------------

    def _build_waypoints(self):
        """Returns list of (x, y, speed) tuples describing the full path."""
        wps = []
        speed = self._travel_speed  # 2.68 m/s constant
        n_line = 30   # waypoints per straight segment
        n_arc = 18    # waypoints per quarter circle arc

        # Start at (0, 3)
        current = (0.0, 3.0)

        # Segment 1: Straight from (0, 3) to (0, 7)
        target = (0.0, 7.0)
        for i in range(1, n_line + 1):
            t = i / n_line
            x = current[0] + t * (target[0] - current[0])
            y = current[1] + t * (target[1] - current[1])
            wps.append((x, y, speed))
        current = target

        # Segment 2: Three-quarter circle around (3, 7) clockwise from (0, 7) to (3, 4), radius 3 (270°)
        cx, cy, r = 3.0, 7.0, 3.0
        start_angle = np.pi  # (0, 7) relative to (3, 7) is at 180°
        for i in range(1, 3 * n_arc + 1):
            theta = start_angle - i * 3 * np.pi / 2 / (3 * n_arc)  # clockwise 270°
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            wps.append((x, y, speed))
        current = (3.0, 4.0)

        # Segment 3: Straight from (3, 4) to (-4, 4)
        target = (-4.0, 4.0)
        for i in range(1, n_line + 1):
            t = i / n_line
            x = current[0] + t * (target[0] - current[0])
            y = current[1] + t * (target[1] - current[1])
            wps.append((x, y, speed))
        current = target

        # Segment 4: Quarter circle (center -4, 2; radius 2) from (-4, 4) to (-6, 2)
        cx, cy, r = -4.0, 2.0, 2.0
        start_angle = np.pi / 2  # (-4, 4) relative to (-4, 2) is at 90°
        for i in range(1, n_arc + 1):
            theta = start_angle + i * np.pi / 2 / n_arc
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            wps.append((x, y, speed))
        current = (-6.0, 2.0)

        # Segment 5: Straight from (-6, 2) to (-6, -2)
        target = (-6.0, -2.0)
        for i in range(1, n_line + 1):
            t = i / n_line
            x = current[0] + t * (target[0] - current[0])
            y = current[1] + t * (target[1] - current[1])
            wps.append((x, y, speed))
        current = target

        # Segment 6: Quarter circle (center -4, -2; radius 2) anticlockwise from (-6, -2) to (-4, -4)
        cx, cy, r = -4.0, -2.0, 2.0
        start_angle = np.pi  # (-6, -2) relative to (-4, -2) is at 180°
        for i in range(1, n_arc + 1):
            theta = start_angle + i * np.pi / 2 / n_arc
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            wps.append((x, y, speed))
        current = (-4.0, -4.0)

        # Segment 7: Straight from (-4, -4) to (10, 4)
        target = (10.0, 4.0)
        for i in range(1, n_line + 1):
            t = i / n_line
            x = current[0] + t * (target[0] - current[0])
            y = current[1] + t * (target[1] - current[1])
            wps.append((x, y, speed))

        return wps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._idx >= len(self._waypoints)

    def get_velocity(self, state) -> tuple:
        """
        Given the evader's current (x, y) state, return the commanded (vx, vy).
        Call this once per simulation step.
        """
        if self.done:
            return 0.0, 0.0

        wx, wy, speed = self._waypoints[self._idx]

        dx = wx - state[0]
        dy = wy - state[1]
        dist = np.hypot(dx, dy)

        # Advance waypoint when close enough
        if dist < self._REACH_THRESHOLD:
            self._idx += 1
            if self.done:
                return 0.0, 0.0
            wx, wy, speed = self._waypoints[self._idx]
            dx = wx - state[0]
            dy = wy - state[1]
            dist = np.hypot(dx, dy)

        if dist < 1e-9 or speed == 0.0:
            return 0.0, 0.0

        vx = (dx / dist) * speed
        vy = (dy / dist) * speed
        return vx, vy
