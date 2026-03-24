# main.py
import pygame
from models.planar_quadrotor import TopDownQuadrotor
from models.evader import Evader
from controllers.basic_tracker import basic_chase_controller
from visualization.renderer import PygameRenderer

def main():
    dt = 0.01 
    
    # Initialize components
    drone = TopDownQuadrotor(x=0.0, y=0.0, mass=1.0, I_zz=0.02)
    evader = Evader(x=5.0, y=5.0)
    renderer = PygameRenderer()
    
    # Clock to manage simulation speed
    clock = pygame.time.Clock()

    print("Simulation starting. Use W/A/S/D to move the evader.")
    print("Close the window or press Q to quit.")

    running = True
    while running:
        # 1. Handle Window Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
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

        # 5. Render
        renderer.draw(drone.state, evader.state)
        
        # 6. Enforce timestep (runs the loop at roughly 1/dt frames per second)
        clock.tick(int(1 / dt))

    # Clean up and close window
    renderer.quit()

if __name__ == "__main__":
    main()