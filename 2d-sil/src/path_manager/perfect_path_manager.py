from path_manager import PathManager, PathOutput
import numpy as np
import casadi as ca
from track import Track

G = 9.81

class PerfectPathManager(PathManager):
    def __init__(self, m, mu, max_speed_ms, max_accel, max_brake):
        self.m = m
        self.mu = mu
        self.max_speed_ms = max_speed_ms
        self.max_accel = max_accel
        self.max_brake = max_brake
        self.cached_path = None
        self.cached_track_id = None

    @property
    def name(self):
        return "Perfect"

    def compute_path(self, x, y, heading, speed, track: Track):
        if self.cached_track_id != id(track):
            self.cached_path = self._solve(track)
            self.cached_track_id = id(track)

        cl = self.cached_path.waypoints # type: ignore
        dists = np.linalg.norm(cl - np.array([x, y]), axis=1)
        nearest = np.argmin(dists)
        N = len(cl)
        indices = [(nearest + i) % N for i in range(50)]

        return PathOutput(
            waypoints=cl[indices],
            target_speeds=self.cached_path.target_speeds[indices] # type: ignore
        )

    def _solve(self, track: Track):
        N = int(track.num_points)

        cx = track.points["x"]
        cy = track.points['y']
        nx = -np.sin(track.points['heading'])
        ny = np.cos(track.points['heading'])
        half_width = track.points['width'] / 2.0

        alpha = ca.MX.sym('alpha', N)  # type: ignore
        v = ca.MX.sym('v', N)  # type: ignore

        px = cx + alpha * half_width * nx
        py = cy + alpha * half_width * ny

        px_next = ca.vertcat(px[1:], px[0])  # type: ignore
        py_next = ca.vertcat(py[1:], py[0])  # type: ignore
        px_prev = ca.vertcat(px[-1], px[:-1])  # type: ignore
        py_prev = ca.vertcat(py[-1], py[:-1])  # type: ignore

        ds = ca.sqrt((px_next - px)**2 + (py_next - py)**2)

        ux = px - px_prev;  uy = py - py_prev
        vx = px_next - px_prev;  vy = py_next - py_prev
        wx = px_next - px;  wy = py_next - py

        cross = ux * vy - vx * uy
        a_len = ca.sqrt(ux**2 + uy**2)
        b_len = ca.sqrt(wx**2 + wy**2)
        c_len = ca.sqrt(vx**2 + vy**2)
        kappa = 2 * cross / (a_len * b_len * c_len + 1e-9)

        T = ca.sum1(ds / v)

        v_next = ca.vertcat(v[1:], v[0])  # type: ignore
        a_lat = v**2 * kappa
        a_lon = (v_next**2 - v**2) / (2 * ds + 1e-6)

        g_friction = a_lat**2 + a_lon**2

        g_accel = a_lon   # Upper bounded by a_max
        g_brake = a_lon   # Lower bounded by -brake_max

        g = ca.vertcat(g_friction, g_accel, g_brake)

        mu_g = self.mu * G

        opt_vars = ca.vertcat(alpha, v)

        lbx = np.concatenate([-np.ones(N), np.ones(N) * 1e-6])
        ubx = np.concatenate([np.ones(N), np.ones(N) * self.max_speed_ms])

        lbg = np.concatenate([
            np.zeros(N),
            np.full(N, -np.inf),
            np.full(N, -self.max_brake),
        ])
        ubg = np.concatenate([
            np.full(N, mu_g**2),
            np.full(N, self.max_accel),
            np.full(N, np.inf),
        ])

        nlp = {'x': opt_vars, 'f': T, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 300,
            'print_time': 0,
        })

        x0 = np.concatenate([np.zeros(N), np.ones(N) * 10.0])

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        sol_vals = sol['x'].full().flatten()
        alpha_opt = sol_vals[:N]
        v_opt = sol_vals[N:]

        px_opt = cx + alpha_opt * half_width * nx
        py_opt = cy + alpha_opt * half_width * ny
        waypoints = np.column_stack((px_opt, py_opt))

        return PathOutput(waypoints=waypoints, target_speeds=v_opt)
