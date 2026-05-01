import math
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    radius: float


@dataclass
class EllipseObstacle:
    cx: float
    cy: float
    rx: float        # semi-axis along local x (before rotation)
    ry: float        # semi-axis along local y (before rotation)
    theta: float = 0.0  # rotation angle in radians (CCW from world x-axis)


Obstacle = Union[CircleObstacle, EllipseObstacle]


@dataclass
class BaseEnvironment:
    obstacles: List[Obstacle] = field(default_factory=list)

    def check_collision(self, x, y, agent_radius=0.5) -> bool:
        for obs in self.obstacles:
            if isinstance(obs, EllipseObstacle):
                if _point_in_ellipse(x, y, obs, margin=agent_radius):
                    return True
            else:
                if math.hypot(x - obs.cx, y - obs.cy) < obs.radius + agent_radius:
                    return True
        return False

    def has_line_of_sight(self, x1, y1, x2, y2) -> bool:
        for obs in self.obstacles:
            if isinstance(obs, EllipseObstacle):
                if _segment_intersects_ellipse(x1, y1, x2, y2, obs):
                    return False
            else:
                if _segment_intersects_circle(x1, y1, x2, y2, obs.cx, obs.cy, obs.radius):
                    return False
        return True

    def los_clearance(self, x1, y1, x2, y2) -> float:
        """Minimum signed clearance between the LOS segment and any obstacle.

        Positive = the segment clears all obstacles by at least this distance.
        Negative = the segment is occluded (penetrates an obstacle by this depth).
        Returns +inf when there are no obstacles.
        """
        if not self.obstacles:
            return math.inf
        min_clearance = math.inf
        for obs in self.obstacles:
            if isinstance(obs, EllipseObstacle):
                clearance = _segment_clearance_ellipse(x1, y1, x2, y2, obs)
            else:
                clearance = _segment_clearance_circle(x1, y1, x2, y2, obs.cx, obs.cy, obs.radius)
            min_clearance = min(min_clearance, clearance)
        return min_clearance


def _point_in_ellipse(x, y, obs: EllipseObstacle, margin: float = 0.0) -> bool:
    ct, st = math.cos(obs.theta), math.sin(obs.theta)
    dx = x - obs.cx
    dy = y - obs.cy
    lx =  ct * dx + st * dy
    ly = -st * dx + ct * dy
    return (lx / (obs.rx + margin)) ** 2 + (ly / (obs.ry + margin)) ** 2 < 1.0


def _segment_intersects_ellipse(x1, y1, x2, y2, obs: EllipseObstacle) -> bool:
    """Transform segment into the ellipse's normalized unit-circle frame, then check."""
    ct, st = math.cos(obs.theta), math.sin(obs.theta)

    def to_norm(x, y):
        dx = x - obs.cx
        dy = y - obs.cy
        lx = ( ct * dx + st * dy) / obs.rx
        ly = (-st * dx + ct * dy) / obs.ry
        return lx, ly

    p1 = to_norm(x1, y1)
    p2 = to_norm(x2, y2)
    return _segment_intersects_circle(p1[0], p1[1], p2[0], p2[1], 0.0, 0.0, 1.0)


def _segment_intersects_circle(x1, y1, x2, y2, cx, cy, r) -> bool:
    """Closest-point-on-segment test for a circle."""
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def _segment_min_dist_to_point(x1, y1, x2, y2, px, py) -> float:
    """Minimum distance from point (px,py) to the segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _segment_clearance_circle(x1, y1, x2, y2, cx, cy, r) -> float:
    """Signed clearance: distance from segment to circle edge (negative if intersecting)."""
    return _segment_min_dist_to_point(x1, y1, x2, y2, cx, cy) - r


def _segment_clearance_ellipse(x1, y1, x2, y2, obs: EllipseObstacle) -> float:
    """Signed clearance in the ellipse's normalized unit-circle frame."""
    ct, st = math.cos(obs.theta), math.sin(obs.theta)

    def to_norm(x, y):
        dx = x - obs.cx
        dy = y - obs.cy
        return (ct * dx + st * dy) / obs.rx, (-st * dx + ct * dy) / obs.ry

    p1 = to_norm(x1, y1)
    p2 = to_norm(x2, y2)
    return _segment_min_dist_to_point(p1[0], p1[1], p2[0], p2[1], 0.0, 0.0) - 1.0
