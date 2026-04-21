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

    def _draw_grid(self):
        """Draws thin grey gridlines every 10 metres."""
        grid_color = (200, 200, 200)
        grid_spacing_m = 10  # metres
        grid_spacing_px = grid_spacing_m * self.scale

        # Vertical lines
        start_x = self.offset_x % grid_spacing_px
        x = start_x
        while x <= self.width:
            pygame.draw.line(self.screen, grid_color, (int(x), 0), (int(x), self.height))
            x += grid_spacing_px

        # Horizontal lines
        start_y = self.offset_y % grid_spacing_px
        y = start_y
        while y <= self.height:
            pygame.draw.line(self.screen, grid_color, (0, int(y)), (self.width, int(y)))
            y += grid_spacing_px

    def _draw_obstacles(self, env):
        """Draws all obstacles in the environment."""
        for obs in env.obstacles:
            if hasattr(obs, 'rx'):
                self._draw_ellipse_obstacle(obs)
            else:
                cx, cy = self._world_to_screen(obs.cx, obs.cy)
                radius_px = int(obs.radius * self.scale)
                pygame.draw.circle(self.screen, (80, 80, 80), (cx, cy), radius_px)
                pygame.draw.circle(self.screen, (40, 40, 40), (cx, cy), radius_px, 2)

    def _draw_ellipse_obstacle(self, obs):
        ct, st = math.cos(obs.theta), math.sin(obs.theta)
        n_pts = 48
        pts = []
        for i in range(n_pts):
            a = 2 * math.pi * i / n_pts
            lx = obs.rx * math.cos(a)
            ly = obs.ry * math.sin(a)
            wx = obs.cx + ct * lx - st * ly
            wy = obs.cy + st * lx + ct * ly
            pts.append(self._world_to_screen(wx, wy))
        pygame.draw.polygon(self.screen, (80, 80, 80), pts)
        pygame.draw.polygon(self.screen, (40, 40, 40), pts, 2)

    def _draw_los(self, drone_world, evader_world, env):
        """Draws a dotted line from drone to evader, green if clear, red if occluded."""
        has_los = env.has_line_of_sight(
            drone_world[0], drone_world[1],
            evader_world[0], evader_world[1]
        )
        color = (0, 180, 0) if has_los else (220, 0, 0)

        dsx, dsy = self._world_to_screen(drone_world[0], drone_world[1])
        esx, esy = self._world_to_screen(evader_world[0], evader_world[1])

        dx = esx - dsx
        dy = esy - dsy
        length = math.hypot(dx, dy)
        if length == 0:
            return

        dash, gap = 6, 6
        step = dash + gap
        num_steps = int(length / step)
        for i in range(num_steps):
            t_start = i * step / length
            t_end = min((i * step + dash) / length, 1.0)
            p1 = (dsx + dx * t_start, dsy + dy * t_start)
            p2 = (dsx + dx * t_end, dsy + dy * t_end)
            pygame.draw.line(self.screen, color, p1, p2, 1)

    def _draw_collision_message(self, msg):
        """Renders a collision message in the top-right corner."""
        font = pygame.font.SysFont(None, 32)
        text = font.render(msg, True, (220, 0, 0))
        rect = text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(text, rect)

        sub_font = pygame.font.SysFont(None, 24)
        sub = sub_font.render("Press Q to quit", True, (150, 0, 0))
        sub_rect = sub.get_rect(topright=(self.width - 10, rect.bottom + 4))
        self.screen.blit(sub, sub_rect)

    def draw(self, drone_state, evader_state, env=None, collision_msg=None):
        # Fill the background with white
        self.screen.fill((255, 255, 255))

        # Draw gridlines
        self._draw_grid()

        # Draw obstacles
        if env is not None:
            self._draw_obstacles(env)

        # Draw line-of-sight
        if env is not None:
            self._draw_los(
                (drone_state[0], drone_state[1]),
                evader_state,
                env
            )

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

        # Draw collision overlay if needed
        if collision_msg is not None:
            self._draw_collision_message(collision_msg)

        # Update the full display Surface to the screen
        pygame.display.flip()
        
    def quit(self):
        pygame.quit()