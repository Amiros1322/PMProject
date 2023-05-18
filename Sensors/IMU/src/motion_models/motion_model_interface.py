from typing import Union
from abc import ABCMeta, abstractmethod
import numpy as np

class ILinearMotionModel(metaclass=ABCMeta):
    @abstractmethod
    def F(self) -> np.ndarray:
        pass

    @abstractmethod
    def B(self) -> np.ndarray:
        pass


class INonLinearMotionModel(metaclass=ABCMeta):
    @abstractmethod
    def f(self, x, u) -> np.ndarray:
        pass


IMotionModel = Union[ILinearMotionModel, INonLinearMotionModel]
