from utils.geometry import CircleObstacle, EllipseObstacle


class SimpleEnvironment:
    def __init__(self, obstacles):
        self.obstacles = obstacles


def make_mixed_obstacles_env():
    obstacles = [
        CircleObstacle(cx=5.0, cy=5.0, radius=2.0, inflate=0.3),
        EllipseObstacle(cx=11.0, cy=7.5, a=3.2, b=1.6, inflate=0.3),
    ]
    return SimpleEnvironment(obstacles)
