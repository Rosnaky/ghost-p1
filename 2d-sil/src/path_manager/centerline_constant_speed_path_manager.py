from path_manager import PathManager, PathOutput
import numpy as np

class CenterlineConstantSpeedPathManager(PathManager):
    def __init__(self, lookahead_metres=10, base_speed=10.0):
        self.lookahead_metres = lookahead_metres
        self.base_speed = base_speed

    @property
    def name(self):
        return "Centerline"

    def compute_path(self, x, y, heading, speed, track):
        cl = track.centerline()
        dists = np.linalg.norm(cl - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(dists)

        # Grab the next lookahead_metres points
        waypoints = [cl[nearest_idx]]
        accumulated_dist = 0.0
        idx = nearest_idx

        for i in range(1, len(cl)):
            next_idx = (nearest_idx + i) % len(cl)
            seg_len = np.linalg.norm(cl[next_idx] - cl[idx])
            accumulated_dist += seg_len
            waypoints.append(cl[next_idx])
            idx = next_idx
            if accumulated_dist >= self.lookahead_metres:
                break

        waypoints = np.array(waypoints)

        speeds = np.full(len(waypoints), self.base_speed)

        return PathOutput(waypoints=waypoints, target_speeds=speeds)
