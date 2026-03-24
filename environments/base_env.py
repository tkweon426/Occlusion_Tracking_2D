# environments/base_env.py
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    radius: float


@dataclass
class BaseEnvironment:
    obstacles: List[CircleObstacle] = field(default_factory=list)

    def check_collision(self, x, y, agent_radius=0.5) -> bool:
        """Returns True if the agent overlaps any obstacle."""
        for obs in self.obstacles:
            dist = math.hypot(x - obs.cx, y - obs.cy)
            if dist < obs.radius + agent_radius:
                return True
        return False

    def has_line_of_sight(self, x1, y1, x2, y2) -> bool:
        """Returns True if the segment (x1,y1)-(x2,y2) does not pass through any obstacle."""
        for obs in self.obstacles:
            if _segment_intersects_circle(x1, y1, x2, y2, obs.cx, obs.cy, obs.radius):
                return False
        return True


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
