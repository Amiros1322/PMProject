from typing import List, Dict
import numpy as np
from scipy.linalg import block_diag

from src.motion_models.kbm import KinematicBicycleModel
from src.measurement_models.measurement_model_interface import INonLinearMeasurementModel

from src.common import State, Measurement
from src.runge_kutta4 import propagate_state
from src.unscented_transform import UTParams, UTUtils
from src.utils import Utils


class KinematicBicycleModelKF:
    def __init__(self, motion_model: KinematicBicycleModel,
                 measurement_model_map: Dict[str, INonLinearMeasurementModel], initial_state: State, Q: np.ndarray):
        self._augmented_ut_params = UTParams(alpha=0.5, n=motion_model.augmented_state_size)
        self._ut_params = UTParams(alpha=0.5, n=motion_model.state_size)
        self._motion_model = motion_model
        self._measurement_model_map = measurement_model_map
        assert Utils.is_column_vector(initial_state.x)
        self._x = initial_state.x
        assert Utils.is_symmetric(initial_state.P) and Utils.is_positive_definite(initial_state.P)
        self._P = initial_state.P
        self._S = np.linalg.cholesky(initial_state.P)       # S*S.T = P!!
        assert Utils.is_symmetric(Q) and Utils.is_positive_definite(Q)
        self._Sq = np.linalg.cholesky(Q)
        self._delta = 0

    def predict(self, dt: float, delta: float):
        self._delta = delta
        self._non_linear_square_root_predict(dt, delta)
        self._P = None

    def update(self, measurements: List[Measurement] = []):
        if len(measurements) == 0:
            return
        elif len(measurements) == 1:
            self._square_root_update(measurements[0])
        else:
            self._square_root_information_update(measurements)

    def get_state(self) -> State:
        if self._P is None:
            self._P = np.dot(self._S, self._S.T)
        return State(self._x, self._P)

    def _non_linear_square_root_predict(self, dt: float, delta: float):
        xa = np.vstack((self._x, np.zeros((2, 1))))
        Sa = block_diag(self._S, self._Sq)
        X = UTUtils.create_sigma_points(xa, Sa, self._augmented_ut_params.gamma)
        X_predicted = propagate_state(f=self._motion_model.f, dt=dt, x=X, u=delta)
        X_predicted = X_predicted[:self._motion_model.state_size, :]
        self._x = UTUtils.weighted_sum(
            X_predicted,
            self._augmented_ut_params.mean_weight_vector)
        self._S = UTUtils.estimate_covariance_cholesky_decomposition(
            X_predicted, self._x,
            self._augmented_ut_params.covariance_weight_vector)

    def _square_root_update(self, measurement: Measurement):
        X = UTUtils.create_sigma_points(self._x, self._S, self._ut_params.gamma)
        Z = self._measurement_model_map[measurement.source].h(X, self._delta)
        z_hat = UTUtils.weighted_sum(Z, self._ut_params.mean_weight_vector)
        Sz = UTUtils.estimate_covariance_cholesky_decomposition(
            Z, z_hat,
            self._ut_params.covariance_weight_vector,
            np.linalg.cholesky(measurement.R))
        Pxz = UTUtils.weighted_cross_correlation(
            X, self._x, Z, z_hat, self._ut_params.covariance_weight_vector)
        K = UTUtils.forward_slash(Sz, UTUtils.forward_slash(Sz.T, Pxz))
        self._x += np.dot(K, measurement.z - z_hat)
        U = np.dot(K, Sz)
        self._S = UTUtils.cholupdate(self._S, U, -1)
        self._P = np.dot(self._S, self._S.T)

    def _square_root_information_update(self, measurements: List[Measurement]):
        y = UTUtils.back_slash(self._S.T, UTUtils.back_slash(self._S, self._x))
        _, Sy = np.linalg.qr(UTUtils.back_slash(self._S, np.identity(self._motion_model.state_size)))
        X = UTUtils.create_sigma_points(self._x, self._S, self._ut_params.gamma)
        phi = np.zeros_like(y)
        for measurement in measurements:
            Z = self._measurement_model_map[measurement.source].h(X, self._delta)
            z_hat = UTUtils.weighted_sum(Z, self._ut_params.mean_weight_vector)
            Pxz = UTUtils.weighted_cross_correlation(
                X, self._x, Z, z_hat, self._ut_params.covariance_weight_vector)
            Sn = np.linalg.cholesky(measurement.R)
            U = UTUtils.forward_slash(Sn.T, UTUtils.back_slash(self._S.T, UTUtils.back_slash(self._S, Pxz)))
            phi += np.dot(UTUtils.forward_slash(Sn, U), measurement.z - z_hat + np.dot(Pxz.T, y))
            Sy = UTUtils.cholupdate(Sy, U, 1)
        y += phi
        _, self._S = np.linalg.qr(UTUtils.back_slash(Sy, np.identity(self._motion_model.state_size)))
        self._P = np.dot(self._S, self._S.T)
        self._x = np.dot(self._P, y)
