# environments/single_circle.py
from environments.base_env import BaseEnvironment, CircleObstacle

def make_single_circle_env():
    return BaseEnvironment(obstacles=[
        CircleObstacle(cx=3.0, cy=8.0, radius=1.0)
    ])
