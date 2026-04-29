import numpy as np
from kart import KartState
from .types import IMUMeasurement

class SILIMUSensor:
    def __init__(self, num_magnets, update_rate, wheel_base, seed=69, accel_noise=0.03, gyro_noise=0.001, drift_rate=0.0001):
        self.num_magnets = num_magnets
        self.update_rate = update_rate
        self.wheel_base = wheel_base
        self.rng = np.random.default_rng(seed)
        self.accel_noise_std = accel_noise * np.sqrt(update_rate)
        self.gyro_noise_std = gyro_noise * np.sqrt(update_rate)
        self.drift_rate = drift_rate
        self.last_time = -1.0
        self.prev_speed = 0

        self.accel_bias_x = 0
        self.accel_bias_y = 0
        self.gyro_bias = 0

    def measure(self, state: KartState, t_s: float) -> IMUMeasurement | None:
        if t_s - self.last_time < 1.0 / self.update_rate:
            return None

        dt = t_s - self.last_time if self.last_time >= 0 else 1.0 / self.update_rate    
        self.last_time = t_s

        yaw_rate = state.speed_ms * np.tan(state.steer_angle_rad) / self.wheel_base 

        accel_x = (state.speed_ms - self.prev_speed) / dt
        accel_y = state.speed_ms * yaw_rate
        gyro_z = yaw_rate
        self.prev_speed = state.speed_ms

        # Drift
        self.accel_bias_x += self.rng.normal(0, self.drift_rate)
        self.accel_bias_y += self.rng.normal(0, self.drift_rate)
        self.gyro_bias += self.rng.normal(0, self.drift_rate * 0.1)

        # Noise
        accel_x += self.accel_bias_x + self.rng.normal(0, self.accel_noise_std)
        accel_y += self.accel_bias_y + self.rng.normal(0, self.accel_noise_std)
        gyro_z += self.gyro_bias + self.rng.normal(0, self.gyro_noise_std)

        return IMUMeasurement(
            accel_x,
            accel_y,
            gyro_z,
            t_s
        )
