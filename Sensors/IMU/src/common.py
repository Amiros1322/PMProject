from typing import NamedTuple
import numpy as np

class State(NamedTuple):
    x: np.ndarray
    P: np.ndarray


class Measurement(NamedTuple):
    z: np.ndarray
    R: np.ndarray
    source: str
