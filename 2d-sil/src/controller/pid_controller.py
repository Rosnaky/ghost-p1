from path_manager import PathOutput
from .pid import PID
import numpy as np

class PIDController:
    def __init__(
        self,
        wheel_base,
        throttle_pid: PID,
        lateral_pid: PID,
        lookahead_m=15.0,
    ) -> None:
        self.wheel_base = wheel_base
        self.lookahead_m = lookahead_m
        self.throttle_pid = throttle_pid
        self.lateral_pid = lateral_pid

        throttle_pid.reset()
        lateral_pid.reset()

    def compute(self, x, y, heading, speed, path: PathOutput, dt):
        dists = np.linalg.norm(path.waypoints - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(dists)
        nearest = path.waypoints[nearest_idx]

        # Errors
        lateral_target = -nearest[0] * np.sin(heading) + nearest[1] * np.cos(heading)
        lateral_measured = -x * np.sin(heading) + y * np.cos(heading)

        # Lookahead
        effective_lookahead = self.lookahead_m + speed * 0.3
        valid = dists > effective_lookahead * 0.5
        if np.any(valid):
            look_idx = np.argmax(valid)
        else:
            look_idx = len(path.waypoints) - 1

        target = path.waypoints[look_idx]
        lx = (target[0] - x) * np.cos(heading) + (target[1] - y) * np.sin(heading)
        ly = -(target[0] - x) * np.sin(heading) + (target[1] - y) * np.cos(heading)
        L = np.sqrt(lx**2 + ly**2)

        if L < 0.01:
            steer_ff = 0.0
        else:
            steer_ff = np.arctan(2.0 * ly * self.wheel_base / (L**2))

        steer_fb = self.lateral_pid.compute(lateral_measured, lateral_target, dt)
        speed_factor = max(speed * 2.5, 1.0)
        steer = steer_ff + steer_fb / speed_factor

        target_speed = path.target_speeds[nearest_idx]
        speed_error = target_speed - speed
        throttle = self.throttle_pid.compute(speed, target_speed, dt)

        return throttle, steer
