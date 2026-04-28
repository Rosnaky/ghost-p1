from path_manager import PathOutput
import numpy as np

class PurePursuitController:
    def __init__(self, lookahead_metres=8.0, wheelbase=1.05):
        self.lookahead_metres = lookahead_metres
        self.wheelbase = wheelbase

    def compute(self, x, y, heading, speed, path: PathOutput):
        dists = np.linalg.norm(path.waypoints - np.array([x, y]), axis=1)
        valid = dists > self.lookahead_metres * 0.5
        if not np.any(valid):
            idx = len(path.waypoints) - 1
        else:
            idx = np.argmax(valid)

        target = path.waypoints[idx]
        target_speed = path.target_speeds[idx]

        dx = target[0] - x
        dy = target[1] - y
        local_x = dx * np.cos(heading) + dy * np.sin(heading)
        local_y = -dx * np.sin(heading) + dy * np.cos(heading)

        L = np.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            steer = 0.0
        else:
            curvature = 2.0 * local_y / (L**2)
            steer = np.arctan(curvature * self.wheelbase)

        throttle = np.clip((target_speed - speed) * 0.5, -1.0, 1.0)

        return throttle, steer
