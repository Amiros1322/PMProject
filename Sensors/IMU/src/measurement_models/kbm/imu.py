import numpy as np
from Sensors.IMU.src.measurement_models.measurement_model_interface import INonLinearMeasurementModel
from Sensors.IMU.src.motion_models.kbm import StateIdx

class IMUMeasurementModel(INonLinearMeasurementModel):
    def __init__(self, wheelbase: float):
        self._wheelbase = wheelbase

    def h(self, X: np.ndarray, delta: float) -> np.ndarray:
        ax = X[StateIdx.a, :]
        ay = np.square(X[StateIdx.v, :]) * np.tan(delta) / self._wheelbase
        omega = X[StateIdx.v, :] * np.tan(delta) / self._wheelbase
        return np.stack([ax, ay, omega])