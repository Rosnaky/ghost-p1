import numpy as np
from track import Track
from kart import KartState


class Stats:
    def __init__(self, track: Track):
        self.track = track
        self.sim_time = 0.0
        self.lap_count = 0
        self.lap_start_time = 0.0
        self.last_lap_time = None
        self.best_lap_time = None
        self.last_s = 0.0
        self.max_speed = 0.0
        self.min_speed = float('inf')
        self._cl = track.centerline()

    def iter(self, state: KartState, dt: float):
        self.sim_time += dt

        if state.speed_ms > self.max_speed:
            self.max_speed = state.speed_ms
        if state.speed_ms < self.min_speed and state.speed_ms > 0.1:
            self.min_speed = state.speed_ms

        dists = np.linalg.norm(self._cl - np.array([state.x, state.y]), axis=1)
        nearest = np.argmin(dists)
        s_now = self.track.points['s'][nearest]

        if self.last_s > self.track.total_length * 0.8 and s_now < self.track.total_length * 0.2:
            self.lap_count += 1
            self.last_lap_time = self.sim_time - self.lap_start_time
            if self.best_lap_time is None or self.last_lap_time < self.best_lap_time:
                self.best_lap_time = self.last_lap_time
            self.lap_start_time = self.sim_time
            self.max_speed = 0.0
            self.min_speed = float('inf')

        self.last_s = s_now

    def hud_lines(self, state: KartState) -> list[str]:
        current = self.sim_time - self.lap_start_time
        lines = [
            f"Speed: {state.speed_ms:.1f} m/s ({state.speed_ms * 3.6:.0f} km/h)",
            f"Lap: {self.lap_count}  Time: {current:.1f}s",
        ]
        if self.last_lap_time is not None:
            lines.append(f"Last: {self.last_lap_time:.2f}s  Best: {self.best_lap_time:.2f}s")
        if self.min_speed < float('inf'):
            lines.append(f"Max: {self.max_speed:.1f} m/s  Min: {self.min_speed:.1f} m/s")
        return lines
