# environments/single_circle.py
from environments.base_env import BaseEnvironment, CircleObstacle


def make_single_circle_env():
    return BaseEnvironment(obstacles=[
        CircleObstacle(cx=5.0, cy=5.0, radius=3.0)
    ])
