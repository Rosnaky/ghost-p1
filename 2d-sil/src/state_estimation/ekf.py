from dataclasses import dataclass

from sensors.types import GPSFixType, GPSMeasurement, HallMeasurement, IMUMeasurement
import numpy as np

@dataclass
class FusedState:
    px: float
    py: float
    vx: float
    vy: float
    heading: float

class EKF:
    def __init__(self):
        # State vector
        self.x = np.zeros(6) # [px, py, vx, vy, heading, gyro_bias]
        self.P = np.diag([1.0, 1.0, 0.5, 0.5, 0.1, 0.01]) # Covariance
        self.Q = np.diag([0.01, 0.01, 0.5, 0.5, 0.1, 0.0001]) # Process noise

    def predict(self, imu: IMUMeasurement, dt: float):
        px, py, vx, vy, heading, gyro_bias = self.x

        # x_n+1 = Fx + Gu
        px_new = px + vx * dt
        py_new = py + vy * dt

        ax_world = imu.accel_x * np.cos(heading) - imu.accel_y * np.sin(heading)
        ay_world = imu.accel_x * np.sin(heading) + imu.accel_y * np.cos(heading)

        vx_new = vx + ax_world * dt
        vy_new = vy + ay_world * dt

        heading_new = heading + (imu.gyro_z - gyro_bias) * dt

        gyro_bias_new = gyro_bias

        self.x = np.array([px_new, py_new, vx_new, vy_new, heading_new, gyro_bias_new])

        # Build Jacobian
        F = np.eye(6)
        F[0, 2] = dt # px depends on vx
        F[1, 3] = dt # py depends on vy
        F[2, 4] = (-imu.accel_x * np.sin(heading) - imu.accel_y * np.cos(heading)) * dt # vx depends on accel
        F[3, 4] = (imu.accel_x * np.cos(heading) - imu.accel_y * np.sin(heading)) * dt # vy depends on accel
        F[4, 5] = -dt # heading depends on gyro_bias

        self.P = F @ self.P @ F.T + self.Q * dt

    def update_gps(self, gps: GPSMeasurement):
        z = np.array([gps.x, gps.y])

        # Predicted
        h = np.array([self.x[0], self.x[1]])

        # Transformation matrix
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 1] = 1

        R = None
        if gps.fix_type == GPSFixType.RTK:
            R = np.diag((0.014**2, 0.014**2))
        elif gps.fix_type == GPSFixType.STANDARD:
            R = np.diag((2.5**2, 2.5**2))

        self._update_state(z, h, H, R)

    def update_hall(self, hall: HallMeasurement):
        speed = np.sqrt(self.x[2]**2 + self.x[3]**2)
        speed = max(speed, 0.0001)
        
        z = np.array([hall.ground_omega])

        # Predicted
        h = np.array([speed])

        # Transformation matrix
        H = np.zeros((1, 6))
        H[0, 2] = self.x[2] / speed
        H[0, 3] = self.x[3] / speed

        R = np.diag([0.1**2])

        self._update_state(z, h, H, R)

    def get_state(self) -> FusedState:
        return FusedState(
            self.x[0],
            self.x[1],
            self.x[2],
            self.x[3],
            self.x[4],
        )

    
    def _update_state(self, z, h, H, R):
        y = z - h # Residuals
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + K @ y # Update state
        self.P = (np.eye(6) - K @ H) @ self.P # Update covariance
