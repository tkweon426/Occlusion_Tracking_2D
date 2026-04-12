from utils.geometry import CircleObstacle, EllipseObstacle


class SimpleEnvironment:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def check_collision(self, x, y, agent_radius=0.0):
        for obs in self.obstacles:
            aa = obs.a + agent_radius
            bb = obs.b + agent_radius

            dx = x - obs.cx
            dy = y - obs.cy

            if (dx / max(aa, 1e-6)) ** 2 + (dy / max(bb, 1e-6)) ** 2 <= 1.0:
                return True
        return False


def make_mixed_obstacles_env():
    obstacles = [
        CircleObstacle(cx=5.0, cy=5.0, radius=2.0, inflate=0.3),
        EllipseObstacle(cx=-8.0, cy=-2.5, a=3.2, b=1.6, inflate=0.3),
    ]
    return SimpleEnvironment(obstacles)