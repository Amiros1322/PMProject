from abc import ABCMeta, abstractmethod
import numpy as np

class INonLinearMeasurementModel(metaclass=ABCMeta):
    @abstractmethod
    def h(self, x, u) -> np.ndarray:
        pass
