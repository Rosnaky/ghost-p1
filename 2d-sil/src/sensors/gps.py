import numpy as np
from kart import KartState
from .types import GPSMeasurement, GPSFixType

class SILGPSSensor:
    def __init__(self, update_rate, seed=69, rtk_prob=0.9, dropout_prob=0.005):
        self.update_rate = update_rate
        self.rtk_prob = rtk_prob
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)
        self.last_time = -1.0

        self.noise_std = {
            GPSFixType.RTK: 0.014,
            GPSFixType.STANDARD: 2.5,
        }

    def measure(self, state: KartState, t_s: float) -> GPSMeasurement | None:
        if t_s - self.last_time < 1.0 / self.update_rate:
            return None
        self.last_time = t_s

        if self.rng.random() < self.dropout_prob:
            return None

        fix_type = GPSFixType.RTK if self.rng.random() < self.rtk_prob else GPSFixType.STANDARD
        std = self.noise_std[fix_type]

        return GPSMeasurement(
            x=state.x + self.rng.normal(0, std),
            y=state.y + self.rng.normal(0, std),
            fix_type=fix_type,
            timestamp=t_s,
        )
