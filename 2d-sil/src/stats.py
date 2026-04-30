import numpy as np
from track import Track
from kart import KartState
from state_estimation.ekf import FusedState

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

        self.pos_error = 0.0
        self.heading_error = 0.0
        self.speed_error = 0.0

    def iter(self, actual: KartState, fused: FusedState, dt: float):
        self.sim_time += dt

        if actual.speed_ms > self.max_speed:
            self.max_speed = actual.speed_ms
        if actual.speed_ms < self.min_speed and actual.speed_ms > 0.1:
            self.min_speed = actual.speed_ms

        self.pos_error = np.sqrt((actual.x - fused.px)**2 + (actual.y - fused.py)**2)
        self.heading_error = abs((actual.heading_rad - fused.heading + np.pi) % (2 * np.pi) - np.pi)
        self.speed_error = abs(actual.speed_ms - np.sqrt(fused.vx**2 + fused.vy**2))

        dists = np.linalg.norm(self._cl - np.array([actual.x, actual.y]), axis=1)
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
        lines.append(f"EKF err  pos: {self.pos_error:.3f}m  hdg: {np.degrees(self.heading_error):.1f}°  spd: {self.speed_error:.2f}m/s")
        return lines
