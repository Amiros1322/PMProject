from typing import Tuple, NamedTuple, List, Callable
import numpy as np


class State(NamedTuple):
    x: np.ndarray
    P: np.ndarray


class Target(NamedTuple):
    id: int
    is_blue: bool
    existence_probability: float
    state: State


class Localization(NamedTuple):
    x_ego: float
    y_ego: float
    theta_ego: float
    qxx: float
    qyy: float


class Detection(NamedTuple):
    # NOTE: if parameters are changed,
    cam_x: float
    cam_y: float
    cam_z: float
    cone_class: str
    cov_mat: np.ndarray


class Track(NamedTuple):
    last_detection: Detection
    survived_percent: float


class Tracks(NamedTuple):
    detections: List[Track]


class Car:
    # At time t, these are the car's coordinates relative to the car's position at t-1.
    x_ego = 0  #: float
    y_ego = 0  #: float
    theta_ego = 0  #: float
    x_global: float
    y_global: float
    theta_global: float
    fov: float

    # for graphing the fov:
    left_rel_slope: float
    left_intercept: float
    left_angle: float
    right_intercept: float
    right_rel_slope: float
    right_angle: float
    center_slope: float
    center_intercept: float


class SimulatedCone:
    def __init__(self, x=None, y=None, cone_class=None, range=None, angle=None):
        self.x = x
        self.y = y
        self.cone_class = cone_class
        self.range = range
        self.angle = angle

    def __repr__(self):
        return f"(x={self.x}, y={self.y}, class={self.cone_class}, range={self.range}, angle={self.angle})"


class Car:
    def __init__(self, track_func: Callable[[float], Tuple[float, float]], car_t: float, eps: float, car_angle: float,
                 fov: float):
        p_car = track_func(car_t + eps)
        self.x_global = p_car[0]
        self.y_global = p_car[1]
        self.theta_global = car_angle
        self.fov = fov
        self.track_func = track_func

        # At time k, these are the car's coordinates relative to the car's position at k-1.
        self.x_ego = 0  #: float
        self.y_ego = 0  #: float
        self.theta_ego = 0  #: float

        # initializing variables for graphing the fov:
        self._center_slope, self._center_intercept, self._left_rel_slope, self._left_intercept, self._left_angle, \
        self._right_rel_slope, self._right_intercept, self._right_angle = [None] * 8

        # populating the fov variables
        self._update_fov_lines()

    def detect(self, cone_points):
        return cone_points

    def filter_out_of_fov(self, cone_points: List[SimulatedCone]) -> List[SimulatedCone]:
        """
        Detects points based on the fov and returns the points that were detected

        Args:
            cone_points: List[SimulatedCones]

        Returns: List[SimulatedCones]
        """
        assert isinstance(cone_points[0], SimulatedCone)

        # Convert the points to a NumPy array
        points = np.array([[cone.x, cone.y] for cone in cone_points])
        cone_points = np.array(cone_points)

        left_slope = self._left_rel_slope + self._center_slope
        right_slope = self._right_rel_slope + self._center_slope

        # Calculate the y values for each line at the x coordinates of the points
        cone_x_vals = points[:, 0]
        y_left = left_slope * cone_x_vals + self._left_intercept
        y_right = right_slope * cone_x_vals + self._right_intercept

        # Find the points where y is between y_left and y_right
        cone_y_vals = points[:, 1]
        mask = np.logical_and(cone_y_vals >= np.minimum(y_left, y_right), cone_y_vals <= np.maximum(y_left, y_right))

        # Return the points that satisfy the condition
        seen_points = cone_points[mask]
        return seen_points.tolist()

    def _update_fov_lines(self):
        # Calculate the half angle of the field of view
        half_fov = self.fov / 2

        # Calculate the angles of the two lines that form the border of the field of view
        self._left_angle = self.theta_global - half_fov
        self._right_angle = self.theta_global + half_fov

        # Calculate the slope of the center line
        self._center_slope = np.tan(self.theta_global)

        # Calculate the slopes of the two lines relative to the center line
        left_relative_slope = np.tan(self._left_angle - self.theta_global)
        right_relative_slope = np.tan(self._right_angle - self.theta_global)
        self._left_rel_slope, self._right_rel_slope = left_relative_slope, right_relative_slope

        # Calculate the y-intercept of each line
        left_intercept = self.y_global - (left_relative_slope + self._center_slope) * self.x_global
        right_intercept = self.y_global - (right_relative_slope + self._center_slope) * self.x_global
        center_intercept = self.y_global - self._center_slope * self.x_global
        self._left_intercept, self._right_intercept, self._center_intercept = left_intercept, right_intercept, center_intercept

    def move_car(self, localization: Localization) -> None:
        """
        Updates the car's state given localization

        Args:
            localization: Localization
        """

        self.x_ego = localization.x_ego
        self.y_ego = localization.y_ego
        self.theta_ego = localization.theta_ego

        self.theta_global += localization.theta_ego

        # TODO: THese are wrong for sure. Its adding in different reference frames.
        self.x_global += localization.x_ego
        self.y_global += localization.y_ego

        self._update_fov_lines()

    def move_forward(self, distance: float, qxx=0, qyy=0) -> None:
        loc = Localization(x_ego=distance, y_ego=0, theta_ego=0, qxx=qxx, qyy=qyy)
        self.move_car(loc)



def get_detection_state(detection: Detection) -> State:
    x = np.array([detection.cam_x, detection.cam_y, detection.cam_z])
    P = detection.cov_mat
    state = State(x, P)
    return state
