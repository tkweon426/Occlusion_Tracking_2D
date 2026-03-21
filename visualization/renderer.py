# visualization/renderer.py
import pygame
import math

class PygameRenderer:
    def __init__(self, width=800, height=800, scale=30.0):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale # Pixels per meter
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Optimal Tracking Sim")
        
        # Center the physics origin (0,0) in the middle of the screen
        self.offset_x = width // 2
        self.offset_y = height // 2

    def _world_to_screen(self, x, y):
        """Converts physics coordinates to screen pixels."""
        screen_x = int(self.offset_x + x * self.scale)
        # Invert Y because Pygame's origin (0,0) is at the top-left
        screen_y = int(self.offset_y - y * self.scale) 
        return screen_x, screen_y

    def draw(self, drone_state, evader_state):
        # Fill the background with white
        self.screen.fill((255, 255, 255))
        
        # 1. Draw Evader (Red Circle)
        ex, ey = evader_state
        sx, sy = self._world_to_screen(ex, ey)
        pygame.draw.circle(self.screen, (255, 0, 0), (sx, sy), 8)

        # 2. Draw Drone (Blue Triangle showing heading)
        dx, dy, psi = drone_state[0], drone_state[1], drone_state[2]
        dsx, dsy = self._world_to_screen(dx, dy)
        
        # Calculate triangle vertices based on heading (psi)
        size = 15
        p1 = (dsx + size * math.cos(psi), dsy - size * math.sin(psi))
        p2 = (dsx + size * 0.5 * math.cos(psi + 2.5), dsy - size * 0.5 * math.sin(psi + 2.5))
        p3 = (dsx + size * 0.5 * math.cos(psi - 2.5), dsy - size * 0.5 * math.sin(psi - 2.5))
        
        pygame.draw.polygon(self.screen, (0, 0, 255), [p1, p2, p3])

        # Update the full display Surface to the screen
        pygame.display.flip()
        
    def quit(self):
        pygame.quit()