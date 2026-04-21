# environments/single_ellipse.py
import math
from environments.base_env import BaseEnvironment, EllipseObstacle

def make_single_ellipse_env():
    return BaseEnvironment(obstacles=[
        EllipseObstacle(cx=3.0, cy=7.0, rx=1.5, ry=0.7, theta=math.pi / 4)
    ])

def make_two_ellipse_env():
    return BaseEnvironment(obstacles=[
        EllipseObstacle(cx=3.0, cy=7.0, rx=1.5, ry=0.7, theta=math.pi / 4),
        EllipseObstacle(cx=-4.0, cy=5.0, rx=1.0, ry=2.0, theta=0.0),
    ])
