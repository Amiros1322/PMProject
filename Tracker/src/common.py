from typing import Tuple, NamedTuple, List, Callable
import numpy as np


class State(NamedTuple):
    x: np.ndarray
    P: np.ndarray


class Target(NamedTuple):
    id: int
    cone_class: str
    existence_probability: float
    state: State


class Localization(NamedTuple):
    x_ego: float
    y_ego: float
    theta_ego: float
    qxx: float
    qyy: float


class Detection:
    def __init__(self, cam_x: float, cam_y: float, cam_z: float, cone_class: str, cov_mat: np.ndarray):
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z
        self.cone_class = cone_class
        self.cov_mat = cov_mat

    def _asdict(self):
        return {"cam_x": self.cam_x, "cam_y": self.cam_y, "cam_z": self.cam_z,
                "cone_class": self.cone_class, "cov_mat": self.cov_mat}

    def __repr__(self):
        s = ""
        for key, val in self._asdict():
            s += f"{key}:{val} "
        return s


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

    def detect(self, cone_points, filter_fov=False):
        if filter_fov:
            cone_points = self.filter_out_of_fov(cone_points)
        return cone_points

    def filter_out_of_fov(self, cone_points: List[Detection]) -> List[Detection]:
        """
        Detects points based on the fov and returns the points that were detected

        Args:
            cone_points: List[SimulatedCones]

        Returns: List[SimulatedCones]
        """
        if len(cone_points) == 0:
            return []

        assert isinstance(cone_points[0], Detection)

        # Convert the points to a NumPy array
        points = np.array([[cone.cam_x, cone.cam_y] for cone in cone_points])
        cone_points = np.array(cone_points)

        left_slope = self._left_rel_slope + self._center_slope
        right_slope = self._right_rel_slope + self._center_slope

        # Calculate the y values for each line at the x coordinates of the points
        cone_x_vals = points[:, 0]
        y_left = left_slope * cone_x_vals + self._left_intercept
        y_right = right_slope * cone_x_vals + self._right_intercept

        # Find the points where y is between y_left and y_right
        cone_y_vals = points[:, 1]
        fov_angle_mask = np.logical_and(cone_y_vals >= np.minimum(y_left, y_right), cone_y_vals <= np.maximum(y_left, y_right))
        in_front_of_car_mask = self.points_in_front_of_car_mask(points)

        # Return the points that satisfy the condition
        final_mask = np.logical_and(fov_angle_mask, in_front_of_car_mask)
        seen_points = cone_points[final_mask]
        return seen_points.tolist()

    def points_in_front_of_car_mask(self, points):
        points_in_car_ref_frame = points - np.array([self.x_global, self.y_global])
        reference_vector = np.array([1, -1 / self._center_slope]) if self._center_slope != 0 else np.array([0, 1])
        cross_products = np.cross(points_in_car_ref_frame, reference_vector)
        mask = np.where(cross_products > 0, True, False)
        return mask

    def _update_fov_lines(self, ego=False):
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
        y_line = 0 if ego else self.y_global
        x_line = 0 if ego else self.x_global
        left_intercept = y_line - (left_relative_slope + self._center_slope) * x_line
        right_intercept = y_line - (right_relative_slope + self._center_slope) * x_line
        center_intercept = y_line - self._center_slope * x_line
        self._left_intercept, self._right_intercept, self._center_intercept = left_intercept, right_intercept, center_intercept

    def move_car(self, localization: Localization, ego: bool) -> None:
        """
        Updates the car's state given localization

        Args:
            localization: Localization
        """
        assert -2 * np.pi <= localization.theta_ego <= 2 * np.pi

        self.x_ego = localization.x_ego
        self.y_ego = localization.y_ego
        self.theta_ego = localization.theta_ego

        F_car = np.array([[np.cos(self.theta_global), -np.sin(self.theta_global)],
                      [np.sin(self.theta_global), np.cos(self.theta_global)]])
        new_loc = F_car @ np.array([self.x_ego, self.y_ego])

        print(f"OLDY: {self.y_global}, Translation: {new_loc[1]}")
        print(f"OLDX: {self.x_global}, Translation: {new_loc[0]}")
        print(f"Angle: {np.degrees(self.theta_global)}")

        # Important that self.theta_global update be after movement calc. Otherwise hard to know where car will end up
        # when creating localizations.

        # update variables.
        self.theta_global = (self.theta_global + localization.theta_ego) % (2*np.pi)
        self.x_global = self.x_global + new_loc[0]
        self.y_global = self.y_global + new_loc[1]

        self._update_fov_lines(ego=ego)

    def move_forward(self, distance: float, ego, qxx=0, qyy=0) -> None:
        loc = Localization(x_ego=distance, y_ego=0, theta_ego=0, qxx=qxx, qyy=qyy)
        self.move_car(loc, ego)

    def rotate(self, angle: float, ego, degrees=False):
        if degrees:
            angle = np.radians(degrees)
        loc = Localization(x_ego=0, y_ego=0, theta_ego=angle, qxx=1, qyy=1)
        self.move_car(loc, ego)



def get_detection_state(detection: Detection) -> State:
    x = np.array([detection.cam_x, detection.cam_y, detection.cam_z])
    P = detection.cov_mat
    state = State(x, P)
    return state
