# main.py
import pygame
from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from controllers.basic_tracker import basic_chase_controller
from visualization.renderer import PygameRenderer
from environments.single_circle import make_single_circle_env

DRONE_RADIUS  = 0.5  # metres
EVADER_RADIUS = 0.3  # metres

def main():
    dt = 0.01

    # Initialize components
    drone = TopDownQuadrotor(x=0.0, y=0.0, mass=1.0, I_zz=0.02)
    evader = Evader(x=0.0, y=5.0)
    renderer = PygameRenderer()
    env = make_single_circle_env()

    # Clock to manage simulation speed
    clock = pygame.time.Clock()

    print("Simulation starting. Use W/A/S/D to move the evader.")
    print("Close the window or press Q to quit.")

    running = True
    collided = False
    collision_msg = None

    while running:
        # 1. Handle Window Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        if not collided:
            # 2. Handle Continuous Keyboard Input for Evader
            keys = pygame.key.get_pressed()
            vx, vy = 0.0, 0.0

            if keys[pygame.K_w]: vy = evader.speed
            if keys[pygame.K_s]: vy = -evader.speed
            if keys[pygame.K_a]: vx = -evader.speed
            if keys[pygame.K_d]: vx = evader.speed

            # 3. Control Logic for Drone
            drone_action = basic_chase_controller(drone.state, evader.state)

            # 4. Step Physics
            evader.step(vx, vy, dt)
            drone.step(drone_action, dt)

            # 5. Collision Check
            if env.check_collision(drone.state[0], drone.state[1], DRONE_RADIUS):
                collided = True
                collision_msg = "DRONE COLLISION"
            elif env.check_collision(evader.state[0], evader.state[1], EVADER_RADIUS):
                collided = True
                collision_msg = "EVADER COLLISION"

        # 6. Render
        renderer.draw(drone.state, evader.state, env=env, collision_msg=collision_msg)

        # 7. Enforce timestep (runs the loop at roughly 1/dt frames per second)
        clock.tick(int(1 / dt))

    # Clean up and close window
    renderer.quit()

if __name__ == "__main__":
    main()