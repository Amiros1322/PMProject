from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Sensors.IMU.src.runge_kutta4 import propagate_state
from Sensors.IMU.src.motion_models.kbm import KinematicBicycleModel, StateIdx
from Sensors.IMU.src.kalman_filter import KinematicBicycleModelKF
from Sensors.IMU.src.measurement_models.kbm.imu import IMUMeasurementModel
from Sensors.IMU.src.measurement_models.kbm.wheel_speed import WheelSpeedMeasurementModel
from Sensors.IMU.src.measurement_models.kbm.imu_and_wheel_speed import IMUAndWheelSpeedMeasurementModel
from Sensors.IMU.src.common import State, Measurement


MILLI2SEC = 1e-3
IMU_STR = 'IMU'
WHEEL_SPEED_STR = 'WS'
COMBINED_STR = 'IMU_WS'

class KalmanFilterSimulator:
    def __init__(self):
        self._dt = 50 * MILLI2SEC                       # [sec]
        self._t_max = 10                                # [sec]
        self._wheelbase_length = 1.5                    # [m]
        self._tire_radius = 0.2                         # [m]
        self._wheel_speed_noise_std = np.deg2rad(5)     # [rad/sec]
        self._imu_acceleration_noise_std = 0.4          # [m/sec^2]
        self._gyro_noise_std = np.deg2rad(10)           # [rad/sec]
        self._jerk_process_noise_std = 1                # [m/sec^3]
        self._omega_process_noise_std = 0.5             # [rad/sec]
        self._t = np.arange(0, self._t_max + self._dt, self._dt)
        self._motion_model = KinematicBicycleModel(L=self._wheelbase_length)

    def run(self):
        ground_truth_state, delta = self._generate_ground_truth_state()
        wheel_speed = self._generate_wheel_speed_measurements(ground_truth_state)
        imu = self._generate_imu_measurements(ground_truth_state, delta)
        # estimated_trajectory = self._estimate_trajectory_info_update(delta, wheel_speed, imu)
        estimated_trajectory = self._estimate_trajectory_single_update(delta, wheel_speed, imu)
        self._visualize_results(ground_truth_state, estimated_trajectory)

    def _generate_ground_truth_state(self) -> np.ndarray:
        max_steering = np.deg2rad(10)
        time_it_takes_to_turn_the_wheel_left_and_right = 5  # [sec]
        # What delta represents?? - it's a sin function amplitude = (-0.1745, 0.1745)
        delta = max_steering * np.sin(2 * np.pi * self._t / time_it_takes_to_turn_the_wheel_left_and_right)
        acceleration_time = 2
        acceleration_value = 5
        # Is "a" is the acceleration??
        a = np.zeros_like(self._t)
        a[self._t < acceleration_time] = acceleration_value
        a[self._t > (self._t_max - acceleration_time)] = -acceleration_value
        # X:
        # line: length of 201 that represents the k values
        # each row is value of X (x, y, theta, v, a, omega_noise, and jerk_noise)
        X = np.zeros(shape=(self._motion_model.augmented_state_size, len(self._t)))
        # If it already 0.0 why should we do it again?
        X[StateIdx.x, 0] = 0
        X[StateIdx.y, 0] = 0
        X[StateIdx.theta, 0] = 0
        X[StateIdx.v, 0] = 0
        X[StateIdx.a, 0] = a[0]     # Except this line
        for ii in range(len(self._t) - 1):
            X[:, [ii + 1]] = propagate_state(f=self._motion_model.f, dt=self._dt, x=X[:, [ii]], u=delta[ii])
            X[StateIdx.a, [ii + 1]] = a[ii + 1]
        return X[:5, :], delta
        
    def _generate_wheel_speed_measurements(self, ground_truth_state: np.ndarray) -> np.ndarray:
        wheel_speed = ground_truth_state[StateIdx.v, :] / self._tire_radius
        noise = np.random.normal(loc=0.0, scale=self._wheel_speed_noise_std, size=len(self._t))
        return wheel_speed + noise

    def _generate_imu_measurements(self, ground_truth_state: np.ndarray, delta: np.ndarray) -> np.ndarray:
        X = ground_truth_state
        ax = X[StateIdx.a, :]
        ay = np.square(X[StateIdx.v, :]) * np.tan(delta) / self._wheelbase_length
        omega = X[StateIdx.v, :] * np.tan(delta) / self._wheelbase_length
        noise_ax = np.random.normal(loc=0.0, scale=self._imu_acceleration_noise_std, size=len(self._t))
        noise_ay = np.random.normal(loc=0.0, scale=self._imu_acceleration_noise_std, size=len(self._t))
        noise_omega = np.random.normal(loc=0.0, scale=self._gyro_noise_std, size=len(self._t))
        return np.stack((ax + noise_ax, ay + noise_ay, omega + noise_omega))

    # Debug!!!
    def _estimate_trajectory_info_update(self, delta: np.ndarray, wheel_speed: np.ndarray, imu: np.ndarray) -> np.ndarray:
        measurement_model_map = {
            IMU_STR: IMUMeasurementModel(self._wheelbase_length),
            WHEEL_SPEED_STR: WheelSpeedMeasurementModel(self._tire_radius)
        }
        Q = np.diag([self._omega_process_noise_std ** 2, self._jerk_process_noise_std ** 2])
        initial_state = State(
            x=np.zeros((self._motion_model.state_size, 1)),
            P=np.square(np.diag([0.2, 0.2, 0.2, 0.3, 0.5]))
        )
        kf = KinematicBicycleModelKF(self._motion_model, measurement_model_map, initial_state, Q)
        R_imu = np.diag([
            self._imu_acceleration_noise_std ** 2,
            self._imu_acceleration_noise_std ** 2,
            self._omega_process_noise_std ** 2])
        R_ws = np.array([[self._wheel_speed_noise_std ** 2]])
        kf.update([
            Measurement(z=imu[:, [0]], R=R_imu, source=IMU_STR),
            Measurement(z=wheel_speed[0], R=R_ws, source=WHEEL_SPEED_STR),
            ])
        states = [kf.get_state()]
        for ii in range(1, len(self._t)):           # Main Loop
            kf.predict(self._dt, delta=delta[ii])
            kf.update([
                Measurement(z=imu[:, [ii]], R=R_imu, source=IMU_STR),
                Measurement(z=wheel_speed[ii], R=R_ws, source=WHEEL_SPEED_STR),
            ])
            states.append(kf.get_state())
        return states

    def _estimate_trajectory_single_update(self, delta: np.ndarray, wheel_speed: np.ndarray, imu: np.ndarray) -> np.ndarray:
        measurement_model_map = {COMBINED_STR: IMUAndWheelSpeedMeasurementModel(self._wheelbase_length, self._tire_radius)}
        Q = np.diag([self._omega_process_noise_std ** 2, self._jerk_process_noise_std ** 2])
        initial_state = State(
            x=np.zeros((self._motion_model.state_size, 1)),
            P=np.square(np.diag([0.2, 0.2, 0.2, 0.3, 0.5]))
        )
        kf = KinematicBicycleModelKF(self._motion_model, measurement_model_map, initial_state, Q)
        R = np.diag([
            self._imu_acceleration_noise_std ** 2,
            self._imu_acceleration_noise_std ** 2,
            self._omega_process_noise_std ** 2,
            self._wheel_speed_noise_std ** 2])
        kf.update([Measurement(z=np.append(imu[:, [0]], wheel_speed[0]).reshape(-1, 1), R=R, source=COMBINED_STR)])
        states = [kf.get_state()]
        for ii in range(1, len(self._t)):
            kf.predict(self._dt, delta=delta[ii])
            kf.update([Measurement(z=np.append(imu[:, [ii]], wheel_speed[ii]).reshape(-1, 1), R=R, source=COMBINED_STR)])
            states.append(kf.get_state())
        return states

    def _visualize_results(self, GT: np.ndarray, estimation: List[State]):
        if matplotlib.get_backend() == "agg":
            matplotlib.use("TkAgg")

        X = np.concatenate([state.x for state in estimation], axis=1)
        plt.plot(GT[StateIdx.x, :], GT[StateIdx.y, :], color='blue', label='ground truth')
        plt.plot(X[StateIdx.x, :], X[StateIdx.y, :], color='red', label='estimation')
        plt.legend()
        plt.title(label='Positions')
        plt.show()
        plt.plot(self._t, GT[StateIdx.v, :], color='blue', label='ground truth')
        plt.plot(self._t, X[StateIdx.v, :], color='red', label='estimation')
        plt.legend()
        plt.title(label='Velocities')
        plt.show()
        plt.plot(self._t, GT[StateIdx.a, :], color='blue', label='ground truth')
        plt.plot(self._t, X[StateIdx.a, :], color='red', label='estimation')
        plt.legend()
        plt.title(label='Accelerations')
        plt.show()
        pass


def start_kalman_filter_simulator():
    KalmanFilterSimulator().run()


if __name__ == "__main__":
    start_kalman_filter_simulator()
