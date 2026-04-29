import pygame
import numpy as np
from track import Track
from path_manager import PathOutput


SCREEN_W, SCREEN_H = 1400, 900
BG = (25, 25, 30)
TRACK_FILL = (60, 60, 65)
BOUNDARY = (180, 180, 180)
CENTERLINE = (80, 80, 90)
KART = (0, 200, 100)
PATH = (255, 180, 0)
TRAIL = (100, 100, 255)
HUD = (200, 200, 210)
START = (0, 180, 80)


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Ghost-P1 SIL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.zoom = 8.0
        self.cam_x = 0.0
        self.cam_y = 0.0

    def tick(self):
        return self.clock.tick(60) / 1000.0

    def handle_scroll(self, y):
        self.zoom *= 1.1 if y > 0 else 0.9
        self.zoom = max(1.0, min(50.0, self.zoom))

    def set_camera(self, x, y):
        self.cam_x = x
        self.cam_y = y

    def clear(self):
        self.screen.fill(BG)

    def flip(self):
        pygame.display.flip()

    def _w2s(self, wx, wy):
        return (
            int((wx - self.cam_x) * self.zoom + SCREEN_W / 2),
            int(-(wy - self.cam_y) * self.zoom + SCREEN_H / 2),
        )

    def _polyline(self, points, color, width=2, closed=True):
        pts = [self._w2s(p[0], p[1]) for p in points]
        if len(pts) > 1:
            pygame.draw.lines(self.screen, color, closed, pts, width)

    def draw_track(self, track: Track):
        left = track.left_boundary()
        right = track.right_boundary()
        cl = track.centerline()

        pts_l = [self._w2s(p[0], p[1]) for p in left]
        pts_r = [self._w2s(p[0], p[1]) for p in right]
        polygon = pts_l + pts_r[::-1]
        if len(polygon) > 2:
            pygame.draw.polygon(self.screen, TRACK_FILL, polygon)

        self._polyline(left, BOUNDARY, 2)
        self._polyline(right, BOUNDARY, 2)
        self._polyline(cl, CENTERLINE, 1)

        sl = self._w2s(left[0, 0], left[0, 1])
        sr = self._w2s(right[0, 0], right[0, 1])
        pygame.draw.line(self.screen, START, sl, sr, 3)

    def draw_racing_line(self, waypoints):
        self._polyline(waypoints, PATH, 2)

    def draw_trail(self, trail):
        if len(trail) > 1:
            pts = [self._w2s(p[0], p[1]) for p in trail]
            pygame.draw.lines(self.screen, TRAIL, False, pts, 1)

    def draw_kart(self, x, y, heading):
        L, W = 12, 7
        body = np.array([[L, 0], [-L / 2, W], [-L / 2, -W]], dtype=float)
        c, s = np.cos(-heading), np.sin(-heading)
        R = np.array([[c, -s], [s, c]])
        sx, sy = self._w2s(x, y)
        pts = (R @ body.T).T + np.array([sx, sy])
        pygame.draw.polygon(self.screen, KART, pts.astype(int)) # type: ignore

    def draw_hud(self, lines):
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, HUD)
            self.screen.blit(surf, (10, 10 + i * 22))
