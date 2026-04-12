# args.py
# Central configuration for the simulation.
# Edit this file to change the environment, controller, or any simulation settings.

# --- Environment ---
from environments.mixed_obstacles_env import make_mixed_obstacles_env
ENV_FN = make_mixed_obstacles_env  # callable that returns a BaseEnvironment

# --- Controller ---
from controllers.paper_occlusion_controller import paper_occlusion_controller
CONTROLLER_FN = paper_occlusion_controller
# --- Simulation ---
DT = 0.01                              # timestep (seconds)

# --- Drone ---
DRONE_START    = (0.0, 0.0)           # initial (x, y) in metres
DRONE_MASS     = 1.0                   # kg
DRONE_I_ZZ     = 0.02                  # kg·m²
DRONE_RADIUS   = 0.5                   # collision radius (metres)

# --- Evader ---
EVADER_START   = (0.0, 5.0)           # initial (x, y) in metres
EVADER_RADIUS  = 0.3                   # collision radius (metres)

# --- Renderer ---
RENDERER_WIDTH  = 800                  # window width (pixels)
RENDERER_HEIGHT = 800                  # window height (pixels)
RENDERER_SCALE  = 30.0                 # pixels per metre
