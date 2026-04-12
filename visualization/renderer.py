# visualization/renderer.py
import pygame
import math
from utils.geometry import los_blocked


class PygameRenderer:
    def __init__(self, width=800, height=800, scale=30.0):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Optimal Tracking Sim")

        self.offset_x = width // 2
        self.offset_y = height // 2

    def _world_to_screen(self, x, y):
        screen_x = int(self.offset_x + x * self.scale)
        screen_y = int(self.offset_y - y * self.scale)
        return screen_x, screen_y

    def _draw_grid(self):
        grid_color = (200, 200, 200)
        grid_spacing_m = 10
        grid_spacing_px = grid_spacing_m * self.scale

        start_x = self.offset_x % grid_spacing_px
        x = start_x
        while x <= self.width:
            pygame.draw.line(self.screen, grid_color, (int(x), 0), (int(x), self.height))
            x += grid_spacing_px

        start_y = self.offset_y % grid_spacing_px
        y = start_y
        while y <= self.height:
            pygame.draw.line(self.screen, grid_color, (0, int(y)), (self.width, int(y)))
            y += grid_spacing_px

    def _draw_obstacles(self, env):
        for obs in env.obstacles:
            if getattr(obs, "kind", "circle") == "ellipse":
                left = obs.cx - obs.a
                right = obs.cx + obs.a
                top = obs.cy + obs.b
                bottom = obs.cy - obs.b

                tlx, tly = self._world_to_screen(left, top)
                brx, bry = self._world_to_screen(right, bottom)

                rect_x = tlx
                rect_y = tly
                rect_w = brx - tlx
                rect_h = bry - tly

                rect = pygame.Rect(rect_x, rect_y, rect_w, rect_h)
                pygame.draw.ellipse(self.screen, (80, 80, 80), rect)
                pygame.draw.ellipse(self.screen, (40, 40, 40), rect, 2)
            else:
                cx, cy = self._world_to_screen(obs.cx, obs.cy)
                radius_px = int(obs.a * self.scale)
                pygame.draw.circle(self.screen, (80, 80, 80), (cx, cy), radius_px)
                pygame.draw.circle(self.screen, (40, 40, 40), (cx, cy), radius_px, 2)

    def _draw_los(self, drone_world, evader_world, env):
        blocked = los_blocked(
            (drone_world[0], drone_world[1]),
            (evader_world[0], evader_world[1]),
            env.obstacles,
        )
        color = (0, 180, 0) if not blocked else (220, 0, 0)

        dsx, dsy = self._world_to_screen(drone_world[0], drone_world[1])
        esx, esy = self._world_to_screen(evader_world[0], evader_world[1])

        dx = esx - dsx
        dy = esy - dsy
        length = math.hypot(dx, dy)
        if length == 0:
            return

        dash = 6
        gap = 6
        step = dash + gap
        num_steps = int(length / step)

        for i in range(num_steps + 1):
            t_start = i * step / length
            if t_start > 1.0:
                break
            t_end = min((i * step + dash) / length, 1.0)

            p1 = (dsx + dx * t_start, dsy + dy * t_start)
            p2 = (dsx + dx * t_end, dsy + dy * t_end)
            pygame.draw.line(self.screen, color, p1, p2, 1)

    def _draw_collision_message(self, msg):
        font = pygame.font.SysFont(None, 32)
        text = font.render(msg, True, (220, 0, 0))
        rect = text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(text, rect)

        sub_font = pygame.font.SysFont(None, 24)
        sub = sub_font.render("Press Q to quit", True, (150, 0, 0))
        sub_rect = sub.get_rect(topright=(self.width - 10, rect.bottom + 4))
        self.screen.blit(sub, sub_rect)

    def draw(self, drone_state, evader_state, env=None, collision_msg=None):
        self.screen.fill((255, 255, 255))

        self._draw_grid()

        if env is not None:
            self._draw_obstacles(env)

        if env is not None:
            self._draw_los(
                (drone_state[0], drone_state[1]),
                evader_state,
                env
            )

        ex, ey = evader_state
        sx, sy = self._world_to_screen(ex, ey)
        pygame.draw.circle(self.screen, (255, 0, 0), (sx, sy), 8)

        dx, dy, psi = drone_state[0], drone_state[1], drone_state[2]
        dsx, dsy = self._world_to_screen(dx, dy)

        size = 15
        p1 = (dsx + size * math.cos(psi), dsy - size * math.sin(psi))
        p2 = (dsx + size * 0.5 * math.cos(psi + 2.5), dsy - size * 0.5 * math.sin(psi + 2.5))
        p3 = (dsx + size * 0.5 * math.cos(psi - 2.5), dsy - size * 0.5 * math.sin(psi - 2.5))
        pygame.draw.polygon(self.screen, (0, 0, 255), [p1, p2, p3])

        if collision_msg is not None:
            self._draw_collision_message(collision_msg)

        pygame.display.flip()

    def quit(self):
        pygame.quit()