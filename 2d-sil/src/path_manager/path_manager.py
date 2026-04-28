from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class PathOutput:
    waypoints: np.ndarray
    target_speeds: np.ndarray

class PathManager(ABC):
    @abstractmethod
    def compute_path(self, x, y, heading, speed, track) -> PathOutput:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
