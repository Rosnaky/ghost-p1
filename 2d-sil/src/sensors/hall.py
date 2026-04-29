import numpy as np
from kart import KartState
from .types import HallMeasurement

class SILHallSensor:
    def __init__(self, num_magnets, update_rate, wheel_radius, wheel_base, kart_width, seed=69, noise_pct=2.0):
        self.num_magnets = num_magnets
        self.update_rate = update_rate
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.kart_width = kart_width
        self.noise_pct = noise_pct / 100.0
        self.rng = np.random.default_rng(seed)
        self.last_time = -1.0
        self.speed_quantum = (2 * np.pi * wheel_radius) / num_magnets

    def measure(self, state: KartState, t_s: float) -> HallMeasurement | None:
        if t_s - self.last_time < 1.0 / self.update_rate:
            return None
        self.last_time = t_s

        yaw_rate = state.speed_ms * np.tan(state.steer_angle_rad) / self.wheel_base

        half_w = self.kart_width / 2
        left_speed = state.speed_ms - yaw_rate * half_w
        right_speed = state.speed_ms + yaw_rate * half_w

        true_speeds = np.array([left_speed, right_speed, left_speed, right_speed])
        true_speeds = np.maximum(true_speeds, 0.0)

        wheel_omega = true_speeds / self.wheel_radius

        # Quantize
        quantum_omega = self.speed_quantum / self.wheel_radius
        wheel_omega = np.round(wheel_omega / quantum_omega) * quantum_omega

        # Noise
        noise = self.rng.normal(0, self.noise_pct, 4) * wheel_omega
        wheel_omega += noise

        ground_speed = np.mean(true_speeds)

        return HallMeasurement(
            wheel_omega=wheel_omega,
            ground_omega=float(ground_speed),
            timestamp=t_s,
        )
