import numpy as np


class CircleObstacle:
    def __init__(self, cx, cy, radius, inflate=0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius = float(radius)
        self.a = self.radius + float(inflate)
        self.b = self.radius + float(inflate)
        self.kind = "circle"

    def normalized_level(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        return (dx / self.a) ** 2 + (dy / self.b) ** 2

    def contains_point(self, x, y, margin=0.0):
        aa = self.a + margin
        bb = self.b + margin
        dx = x - self.cx
        dy = y - self.cy
        return (dx / aa) ** 2 + (dy / bb) ** 2 <= 1.0

    def clearance(self, x, y):
        return np.sqrt(max(self.normalized_level(x, y), 1e-12)) - 1.0


class EllipseObstacle:
    def __init__(self, cx, cy, a, b, inflate=0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.a = float(a) + float(inflate)
        self.b = float(b) + float(inflate)
        self.kind = "ellipse"

    def normalized_level(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        return (dx / self.a) ** 2 + (dy / self.b) ** 2

    def contains_point(self, x, y, margin=0.0):
        aa = self.a + margin
        bb = self.b + margin
        dx = x - self.cx
        dy = y - self.cy
        return (dx / aa) ** 2 + (dy / bb) ** 2 <= 1.0

    def clearance(self, x, y):
        return np.sqrt(max(self.normalized_level(x, y), 1e-12)) - 1.0


def point_in_any_obstacle(x, y, obstacles, margin=0.0):
    return any(obs.contains_point(x, y, margin=margin) for obs in obstacles)


def los_blocked(p0, p1, obstacles, n_samples=80):
    x0, y0 = p0
    x1, y1 = p1

    for k in range(n_samples + 1):
        u = k / n_samples
        x = (1.0 - u) * x0 + u * x1
        y = (1.0 - u) * y0 + u * y1
        for obs in obstacles:
            if obs.contains_point(x, y):
                return True
    return False


def min_clearance_to_obstacles(x, y, obstacles):
    if len(obstacles) == 0:
        return np.inf
    return min(obs.clearance(x, y) for obs in obstacles)