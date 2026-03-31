# main.py

import pygame
from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from visualization.renderer import PygameRenderer
import args

def main():
    dt = args.DT

    # Initialize components
    drone = TopDownQuadrotor(x=args.DRONE_START[0], y=args.DRONE_START[1],
                             mass=args.DRONE_MASS, I_zz=args.DRONE_I_ZZ)
    evader = Evader(x=args.EVADER_START[0], y=args.EVADER_START[1])
    renderer = PygameRenderer(width=args.RENDERER_WIDTH, height=args.RENDERER_HEIGHT,
                               scale=args.RENDERER_SCALE)
    env = args.ENV_FACTORY()

    # Clock to manage simulation speed
    clock = pygame.time.Clock()

    scripted = args.EVADER_CONTROLLER  # None → keyboard, object → scripted

    if scripted is None:
        print("Simulation starting. Use W/A/S/D to move the evader.")
    else:
        print("Simulation starting. Evader is following a scripted trajectory.")
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
            # 2. Get evader velocity from keyboard or scripted trajectory
            if scripted is None:
                keys = pygame.key.get_pressed()
                vx, vy = 0.0, 0.0
                if keys[pygame.K_w]: vy = evader.speed
                if keys[pygame.K_s]: vy = -evader.speed
                if keys[pygame.K_a]: vx = -evader.speed
                if keys[pygame.K_d]: vx = evader.speed
            else:
                vx, vy = scripted.get_velocity(evader.state)

            # 3. Control Logic for Drone
            drone_action = args.CONTROLLER(drone.state, evader.state)

            # 4. Step Physics
            evader.step(vx, vy, dt)
            drone.step(drone_action, dt)

            # 5. Collision Check
            if env.check_collision(drone.state[0], drone.state[1], args.DRONE_RADIUS):
                collided = True
                collision_msg = "DRONE COLLISION"
            elif env.check_collision(evader.state[0], evader.state[1], args.EVADER_RADIUS):
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