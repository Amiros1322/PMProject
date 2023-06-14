import numpy as np
from Sensors.IMU.src.measurement_models.measurement_model_interface import INonLinearMeasurementModel
from Sensors.IMU.src.motion_models.kbm import StateIdx

class IMUAndWheelSpeedMeasurementModel(INonLinearMeasurementModel):
    def __init__(self, wheelbase: float, wheel_radius: float):
        self._wheelbase = wheelbase
        self._wheel_radius = wheel_radius

    def h(self, X: np.ndarray, delta: float) -> np.ndarray:
        ax = X[StateIdx.a, :]
        ay = np.square(X[StateIdx.v, :]) * np.tan(delta) / self._wheelbase
        omega = X[StateIdx.v, :] * np.tan(delta) / self._wheelbase
        wheel_speed = X[StateIdx.v, :] / self._wheel_radius
        return np.stack([ax, ay, omega, wheel_speed])