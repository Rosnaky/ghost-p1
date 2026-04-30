from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass(slots=True)
class HallMeasurement:
    wheel_omega: np.ndarray # 4D vector for each wheel clockwise starting from front left
    ground_omega: float # m/s
    timestamp: float # s

@dataclass(slots=True)
class IMUMeasurement:
    accel_x: float # m/s^2
    accel_y: float # m/s^2
    gyro_z: float # rad/s
    timestamp: float # s

class GPSFixType(Enum):
    NONE = 0
    STANDARD = 1
    RTK = 2

@dataclass(slots=True)
class GPSMeasurement:
    x: float # m
    y: float # m
    fix_type: GPSFixType
    timestamp: float # s
