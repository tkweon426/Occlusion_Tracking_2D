# controllers/scripted_evader.py
# Pre-planned trajectory for the evader: orbit obstacle → straight line → decelerate to stop.

import numpy as np

class ScriptedTrajectory:
    """
    A time-indexed waypoint trajectory. The evader is steered toward successive
    waypoints; each waypoint carries a target speed so you get smooth deceleration.

    Phases
    ------
    1. Approach - move from spawn toward the orbit entry point.
    2. Orbit  -  one full counterclockwise loop around the obstacle.
    3. Straight -  travel in a straight line away from the obstacle.
    4. Decelerate - slow to a stop at the destination.
    """

    # How close (metres) the evader must be before it advances to the next waypoint.
    _REACH_THRESHOLD = 0.25

    def __init__(
        self,
        obstacle_cx: float = 5.0,
        obstacle_cy: float = 5.0,
        orbit_radius: float = 4.2,   # clearance around the obstacle edge
        orbit_speed: float = 5.0,    # m/s while looping
        travel_speed: float = 8.0,   # m/s on the straight-line leg
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

        # --- Phase 1: approach to orbit entry (leftmost point of orbit) ---
        entry_angle = np.pi          # angle = 180° → point to the left of the obstacle
        entry = (
            self._cx + self._r * np.cos(entry_angle),
            self._cy + self._r * np.sin(entry_angle),
        )
        wps.append((*entry, self._orbit_speed))

        # --- Phase 2: full counterclockwise loop ---
        # θ goes from π up to π + 2π (counterclockwise in standard math coords)
        for i in range(1, self._n + 1):
            theta = entry_angle + 2 * np.pi * i / self._n
            wx = self._cx + self._r * np.cos(theta)
            wy = self._cy + self._r * np.sin(theta)
            wps.append((wx, wy, self._orbit_speed))

        # --- Phase 3 + 4: straight line with progressive deceleration ---
        ex, ey = entry
        sx, sy = self._stop
        n_decel = 8
        for i in range(1, n_decel + 1):
            t = i / n_decel
            wx = ex + t * (sx - ex)
            wy = ey + t * (sy - ey)
            # Linear speed ramp: full travel_speed → 0
            spd = self._travel_speed * (1.0 - t)
            wps.append((wx, wy, max(spd, 0.0)))

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
