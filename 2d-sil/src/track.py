import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

class Track:
    def __init__(self, points: np.ndarray, total_length: float, is_closed: bool = True):
        self.points = points
        self.total_length = total_length
        self.is_closed = is_closed

    @property
    def num_points(self) -> int:
        return len(self.points)
    
    def centerline(self) -> np.ndarray:
        return np.column_stack((self.points["x"], self.points["y"]))
    
    def left_boundary(self) -> np.ndarray:
        return np.column_stack((self.points["left_x"], self.points["left_y"]))
    
    def right_boundary(self) -> np.ndarray:
        return np.column_stack((self.points["right_x"], self.points["right_y"]))

    def point_at_s(self, s_query: float) -> dict:
        s = self.points["s"]
        if self.is_closed:
            s_query = s_query % self.total_length

        idx = np.searchsorted(s, s_query, side = 'right') - 1
        idx = np.clip(idx, 0, self.num_points - 2)

        s0, s1 = s[idx], s[idx + 1]

        frac = (s_query - s0) / (s1 - s0) if s1 != s0 else 0.0

        result = {}
        for field in self.points.dtype.names: # type: ignore
            v0, v1 = self.points[field][idx], self.points[field][idx + 1]
            if field == "heading":
                # Handle angle wrapping
                diff = (v1 - v0 + np.pi) % (2 * np.pi) - np.pi
                result[field] = v0 + frac * diff
            else:
                result[field] = v0 + frac * (v1 - v0)
        return result

    def curvature_at_s(self, s_query: float) -> float:
        return self.point_at_s(s_query)["curvature"]
    
    def width_at_s(self, s_query: float) -> float:
        return self.point_at_s(s_query)["width"]


def generate_track(seed=None, NUM_SAMPLES=500, TRACK_WIDTH=5.0) -> Track:
    
    rng = np.random.default_rng(seed)

    # Generate constants
    NUM_TRACK_POINTS = rng.integers(low=6, high=12)
    RADIUS_MEAN = rng.uniform(20, 40)
    RADIUS_VARIANCE = rng.uniform(10, 25)

    # Generate points around a circle
    angles = np.sort(rng.uniform(0, 2 * np.pi, NUM_TRACK_POINTS))
    radii = rng.normal(RADIUS_MEAN, RADIUS_VARIANCE, NUM_TRACK_POINTS)
    radii = np.clip(radii, RADIUS_MEAN * 0.4, RADIUS_MEAN * 2.0) # Saturate limits

    # Convert to Cartesian
    cx = radii * np.cos(angles)
    cy = radii * np.sin(angles)

    # Close loop by wrapping endpoints
    cx = np.append(cx, cx[:3])
    cy = np.append(cy, cy[:3])
    t_ctrl = np.arange(len(cx))

    # Fit periodic cubic splines
    cs_x = CubicSpline(t_ctrl, cx, bc_type="periodic") # boundary condition type: periodic so that start = end for derivatives
    cs_y = CubicSpline(t_ctrl, cy, bc_type='periodic')

    # Get samples from spline
    t = np.linspace(0, NUM_TRACK_POINTS, NUM_SAMPLES, endpoint=False)

    x = cs_x(t)
    y = cs_y(t)
    dx = cs_x(t, 1)
    dy = cs_y(t, 1)
    ddx = cs_x(t, 2)
    ddy = cs_y(t, 2)

    heading = np.arctan2(dy, dx)

    speed_sq = dx**2 + dy**2
    curvature = (dx * ddy - dy * ddx) / (speed_sq ** 1.5)

    # Arc length
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # Boundary points
    nx = -np.sin(heading)
    ny = np.cos(heading)
    hw = TRACK_WIDTH / 2.0

    left_x = x + hw * nx
    left_y = y + hw * ny
    right_x = x - hw * nx
    right_y = y - hw * ny

    track_dtype = np.dtype([
        ('x', 'f8'), ('y', 'f8'), ('heading', 'f8'),
        ('curvature', 'f8'), ('width', 'f8'),
        ('left_x', 'f8'), ('left_y', 'f8'),
        ('right_x', 'f8'), ('right_y', 'f8'),
        ('s', 'f8'),
    ])

    points = np.zeros(NUM_SAMPLES, dtype=track_dtype)
    points['x'] = x
    points['y'] = y
    points['heading'] = heading
    points['curvature'] = curvature
    points['width'] = TRACK_WIDTH
    points['left_x'] = left_x
    points['left_y'] = left_y
    points['right_x'] = right_x
    points['right_y'] = right_y
    points['s'] = s

    return Track(points=points, total_length=s[-1])
