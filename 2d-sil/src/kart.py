import numpy as np
from dataclasses import dataclass

@dataclass
class KartState:
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0
    speed_ms: float = 0.0
    steer_angle_rad: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

class KartDynamics:
    def __init__(self, 
                 wheel_base=1.05, 
                 max_steer=np.radians(25), 
                 max_speed=60.0,
                 max_accel=25.0,
                 max_brake=15.0,
                 drag=0.35
    ):
        self.L = wheel_base
        self.max_steer = max_steer
        self.max_speed_ms = max_speed
        self.max_accel = max_accel
        self.max_brake = max_brake
        self.drag = drag

    def step(self, 
             state: KartState, 
             throttle: float,
             steer_cmd: float,
             dt: float,
    ) -> KartState:
        # Saturate inputs
        steer = np.clip(steer_cmd, -self.max_steer, self.max_steer)
        throttle = np.clip(throttle, -1.0, 1.0)

        if throttle >= 0:
            accel = throttle * self.max_accel
        else:
            accel = throttle * self.max_brake

        # Account for drag
        accel -= self.drag * state.speed_ms

        # Update speed
        new_speed_ms = state.speed_ms + accel * dt
        new_speed_ms = np.clip(new_speed_ms, 0.0, self.max_speed_ms)

        # Update position
        if abs(steer) < 1e-6:
            # Straight line
            dx = new_speed_ms * np.cos(state.heading_rad) * dt
            dy = new_speed_ms * np.sin(state.heading_rad) * dt

            dheading = 0.0
        else:
            turn_radius = self.L / np.tan(steer)
            dheading = (new_speed_ms / turn_radius) * dt
            dx = new_speed_ms * np.cos(state.heading_rad + dheading / 2) * dt
            dy = new_speed_ms * np.sin(state.heading_rad + dheading / 2) * dt

        return KartState(
            x=state.x + dx,
            y=state.y + dy,
            heading_rad=state.heading_rad + dheading,
            speed_ms=new_speed_ms,
            steer_angle_rad=steer,
        )
