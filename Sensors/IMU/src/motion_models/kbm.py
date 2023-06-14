from enum import IntEnum
import numpy as np
from Sensors.IMU.src.motion_models.motion_model_interface import INonLinearMotionModel

class StateIdx(IntEnum):
    x = 0
    y = 1
    theta = 2
    v = 3
    a = 4
    omega_noise = 5
    jerk_noise = 6


class KinematicBicycleModel(INonLinearMotionModel):
    def __init__(self, L: float):
        self.L = L      # Axle distance [m]

    def f(self, X: np.ndarray, delta: float) -> np.ndarray:
        """
        This function computes the time derivative of the model at X.
        Here, X is a matrix of states augmented with process noise.
        delta is a deterministic input (scalar) describing the current steering angle
        """
        assert len(X.shape) == 2, 'X Must be a column vector or a matrix'
        assert X.shape[0] == 7, 'X Must have 5 state variables + 2 process noise variables'
        theta = X[StateIdx.theta, :]
        v = X[StateIdx.v, :]
        a = X[StateIdx.a, :]
        omega_process_noise = X[StateIdx.omega_noise, :]
        jerk_process_noise = X[StateIdx.jerk_noise, :]
        X_dot = np.zeros_like(X)
        # Derivatives of X values based on previous k
        X_dot[StateIdx.x, :] = v * np.cos(theta)
        X_dot[StateIdx.y, :] = v * np.sin(theta)
        X_dot[StateIdx.theta, :] = v * np.tan(delta) / self.L + omega_process_noise
        X_dot[StateIdx.v, :] = a
        X_dot[StateIdx.a, :] = jerk_process_noise
        X_dot[StateIdx.omega_noise, :] = 0
        X_dot[StateIdx.jerk_noise, :] = 0
        return X_dot

    @property
    def augmented_state_size(self) -> int:
        return 7

    @property
    def state_size(self) -> int:
        return 5
