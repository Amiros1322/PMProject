import numpy as np
from Sensors.IMU.src.measurement_models.measurement_model_interface import INonLinearMeasurementModel
from Sensors.IMU.src.motion_models.kbm import StateIdx

class WheelSpeedMeasurementModel(INonLinearMeasurementModel):
    def __init__(self, wheel_radius: float):
        self._wheel_radius = wheel_radius

    def h(self, X: np.ndarray, u) -> np.ndarray:
        return X[[StateIdx.v], :] / self._wheel_radius
