import matplotlib.pyplot as plt
from typing import Dict, NamedTuple, List  # TODO: Ask about dataclasses
from Tracker.src.common import Detection, Target
import numpy as np
import copy
from Tracker.src.distance_functions import target_dict_to_np, detection_list_to_np
from Tracker.src.MultiObjectTracker import TrackedCones, MultiObjectTracker
from Tracker.src.common import Localization, Detection, Car, Target


def visualize_consec_frames(prev_cones, curr_cones, x_label, y_label, fontsize=6, color="red", colored_cones=True):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.5)
    visualize_labeling(ax[0], prev_cones, subplot_title="Previous frame cones", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    visualize_labeling(ax[1], curr_cones, subplot_title="Current frame cones", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    plt.show()


def set_axes_options(axes, tick_axes=False):
    """
    Iterates over the axes for options that are always the same.

    Returns: None. Plots on axes
    """
    for ax in axes.flat:
        if tick_axes:
            ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)


def visualize_consec_frames_predictions_detections(prev_cones, curr_cones, prev_dets, curr_dets, prev_pred, curr_pred,
                                                   x_label, y_label, fontsize=6, color="red", colored_cones=True,
                                                   fig_title=None, covariances=False, car=None, ground_truth=None,
                                                   prev_car=None, plot_fov=True, plot_view_angle=False,
                                                   plot_fov_range=(50, 50), fullscreen=False, ego=False):
    fig, ax = plt.subplots(nrows=3, ncols=2 + (ground_truth is not None), figsize=(12, 7), sharex=True, sharey=True)

    if fullscreen:
        fig.canvas.manager.full_screen_toggle()

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plot_tracks(ax[0, 0], prev_cones, subplot_title="Previous tracked cones", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=prev_car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range, ego=ego)
    plot_tracks(ax[0, 1], curr_cones, subplot_title="Current tracked cones", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range, ego=ego)
    plot_tracks(ax[1, 0], prev_pred, subplot_title="Prev Predicted Locations", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=prev_car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range, ego=ego)
    plot_tracks(ax[1, 1], curr_pred, subplot_title="Curr Predicted Locations", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range, ego=ego)
    visualize_labeling(ax[2, 0], prev_dets, subplot_title="Previous Detections", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False, car=prev_car,
                       plot_fov=plot_fov, plot_view_angle=plot_view_angle, plot_fov_range=plot_fov_range, ego=ego)
    visualize_labeling(ax[2, 1], curr_dets, subplot_title="Current Detections", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False, car=car,
                       plot_fov=plot_fov, plot_view_angle=plot_view_angle, plot_fov_range=plot_fov_range, ego=ego)

    if ground_truth is not None:
        ground_truth_axis = fig.add_subplot(1, 3, 3)
        visualize_labeling(ground_truth_axis, ground_truth, subplot_title="Ground truth", x_label=x_label,
                           y_label=y_label,
                           fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False,
                           car_global_x=car.x_global, car_global_y=car.y_global)

    if covariances:
        plot_covariances(ax[0, 0], list(prev_cones.cone_dict.values()))
        plot_covariances(ax[0, 1], list(curr_cones.cone_dict.values()))
        plot_covariances(ax[1, 0], prev_dets)
        plot_covariances(ax[1, 1], curr_dets)

    set_axes_options(ax, tick_axes=True)

    plt.show()


def visualize_consec_frames_with_detections(prev_cones, curr_cones, prev_dets, curr_dets, x_label, y_label, fontsize=6,
                                            color="red", colored_cones=True, fig_title=None, covariances=False,
                                            car=None, ground_truth=None, prev_car=None, plot_fov=True,
                                            plot_view_angle=False,
                                            plot_fov_range=(50, 50), fullscreen=False):
    fig, ax = plt.subplots(nrows=2, ncols=2 + (ground_truth is not None), figsize=(12, 7), sharex=True, sharey=True)

    if fullscreen:
        fig.canvas.manager.full_screen_toggle()

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plot_tracks(ax[0, 0], prev_cones, subplot_title="Previous tracked cones", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=prev_car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range)
    plot_tracks(ax[0, 1], curr_cones, subplot_title="Current tracked cones", x_label=x_label, y_label=y_label,
                fontsize=fontsize, fontcolor=color, car=car, plot_fov=plot_fov, plot_view_angle=plot_view_angle,
                plot_fov_range=plot_fov_range)
    visualize_labeling(ax[1, 0], prev_dets, subplot_title="Previous Detections", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False, car=prev_car,
                       plot_fov=plot_fov, plot_view_angle=plot_view_angle, plot_fov_range=plot_fov_range)
    visualize_labeling(ax[1, 1], curr_dets, subplot_title="Current Detections", x_label=x_label, y_label=y_label,
                       fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False, car=car,
                       plot_fov=plot_fov, plot_view_angle=plot_view_angle, plot_fov_range=plot_fov_range)

    if ground_truth is not None:
        ground_truth_axis = fig.add_subplot(1, 3, 3)
        visualize_labeling(ground_truth_axis, ground_truth, subplot_title="Ground truth", x_label=x_label,
                           y_label=y_label,
                           fontsize=fontsize, fontcolor=color, colored_cones=colored_cones, plot_ids=False,
                           car_global_x=car.x_global, car_global_y=car.y_global)

    if covariances:
        plot_covariances(ax[0, 0], list(prev_cones.cone_dict.values()))
        plot_covariances(ax[0, 1], list(curr_cones.cone_dict.values()))
        plot_covariances(ax[1, 0], prev_dets)
        plot_covariances(ax[1, 1], curr_dets)

    set_axes_options(ax, tick_axes=True)

    plt.show()


def set_axis_titles(ax, subplot_title, x_label, y_label):
    if subplot_title:
        ax.set_title(subplot_title)

    if x_label and y_label:
        ax.set(xlabel=x_label, ylabel=y_label)
    elif x_label:
        ax.set(x_label=x_label)
    elif y_label:
        ax.set(ylabel=y_label)


def plot_car(ax, car, plot_view_angle=False, plot_fov=False, fov_range=(50, 50), ego=False) -> None:
    if car is None:
        return
    assert type(fov_range) == tuple and len(fov_range) == 2

    if ego:
        ax.scatter(0, 0, c="green")
    else:
        ax.scatter(car.x_global, car.y_global, c="green")

    dashed_line_x_start = car.x_global - fov_range[0]
    dashed_line_x_end = car.x_global + fov_range[1]
    if plot_view_angle:
        plot_dashed_line(ax, car._center_slope, car._center_intercept, dashed_line_x_start,
                         dashed_line_x_end, color="blue")
    if plot_fov:
        plot_dashed_line(ax, car._center_slope + car._left_rel_slope, car._left_intercept, dashed_line_x_start,
                         dashed_line_x_end)
        plot_dashed_line(ax, car._center_slope + car._right_rel_slope, car._right_intercept, dashed_line_x_start,
                         dashed_line_x_end, color="purple")


def visualize_ground_truth(ax, cones, car=None, subplot_title="Ground Truth", show_plot=True, plot_car=True,
                           plot_view_angle=False, first_column_color=False):
    """
    Plots cones as they are on the axis.

    Args:
        ax: matplotib axis
        cones: a 2d numpy array representing the cones.
        car: A 'car' object which should have the x_global, y_global, _center_slope, _left_intercept, _left_rel_slope,
             _right_rel_slope, and _right_intercept properties
        subplot_title: (str) The title of the suplot on ax
        show_plot: (bool)  Whether to show the plot immediately after the function or
        plot_car: (bool) Whether to plot the car
        plot_view_angle: (bool) Whether to draw the line coming out the center of the car's fov.
        first_column_color: (bool) Whether the first column of 'cones' has the cone colors

    Returns: None

    """
    x_idx, y_idx, color_idx = (1, 2, 0) if first_column_color else (0, 1, None)
    dashed_line_x_start = cones[:, x_idx].min()
    dashed_line_x_end = cones[:, x_idx].max()
    if car:
        if plot_car:
            ax.scatter(car.x_global, car.y_global, c="green")

        if plot_view_angle:
            plot_dashed_line(ax, car._center_slope, car._center_intercept, dashed_line_x_start,
                             dashed_line_x_end, color="blue")

        plot_dashed_line(ax, car._center_slope + car._left_rel_slope, car._left_intercept, dashed_line_x_start,
                         dashed_line_x_end)
        plot_dashed_line(ax, car._center_slope + car._right_rel_slope, car._right_intercept, dashed_line_x_start,
                         dashed_line_x_end, color="purple")

    if first_column_color:
        ax.scatter(cones[:, x_idx], cones[:, y_idx], color=cones[:, color_idx])
    else:
        ax.scatter(cones[:, x_idx], cones[:, y_idx])

    ax.set_title(subplot_title)
    if show_plot:
        plt.show()


def visualize_labeling(ax, labeled_detections, subplot_title=None, x_label=None, y_label=None, show_plot=False,
                       colored_cones=False, fontsize=6, fontcolor="red", plot_ids=True, plot_conf=False,
                       car=None, plot_view_angle=False, plot_fov=True, plot_fov_range=(50, 50), ego=False):
    """

    Args:
        ax:
        labeled_detections:  Can be either TrackedCones.cone_dict (Dict[int, Targets]) or numpy matrix
        subplot_title:
        x_label:
        y_label:
        show_plot:
        car:
        fontsize:
        color:

    Returns:

    """
    x_idx, y_idx, color_idx = -2, -1, -3
    if isinstance(labeled_detections, Dict):
        labeled_detections = target_dict_to_np(labeled_detections, no_color=not colored_cones)
    elif isinstance(labeled_detections, TrackedCones):
        labeled_detections = target_dict_to_np(labeled_detections.cone_dict, no_color=not colored_cones)
    elif isinstance(labeled_detections, list):
        x_idx, y_idx, color_idx = -3, -2, -4
        labeled_detections, _ = detection_list_to_np(labeled_detections)

    if labeled_detections.size == 0:
        ax.scatter([], [])
    else:
        x_vals, y_vals, colors = labeled_detections[:, x_idx].astype(float).tolist(), \
                                 labeled_detections[:, y_idx].astype(float).tolist(), labeled_detections[:,
                                                                                      color_idx].tolist()
        if colored_cones:
            ax.scatter(x_vals, y_vals, color=colors)
        else:
            assert labeled_detections.dtype in {np.float64, np.float32, np.float16}
            ax.scatter(x_vals, y_vals)

        # iterator over cones
        for i in range(len(x_vals)):
            x = x_vals[i]
            y = y_vals[i]
            ax.text(x, y - 2, f"({round(x)}", fontsize=fontsize, color='purple')

    plot_car(ax, car, plot_view_angle=plot_view_angle, plot_fov=plot_fov, fov_range=plot_fov_range, ego=ego)

    set_axis_titles(ax, subplot_title, x_label, y_label)

    if show_plot:
        plt.show()

def plot_cones_from_dict(ax, cone_dict: Dict[int, Target], fontsize=16, fontcolor="red", plot_ids=True):
    x_vals, y_vals, color_vals = [], [], []
    for t_id, target in cone_dict.items():
        x, y, color = target.state.x[0], target.state.x[1], target.cone_class
        x_vals.append(x)
        y_vals.append(y)
        color_vals.append(color)

        if plot_ids:
            ax.text(x, y, f"id:{t_id} conf:{round(target.existence_probability, 2)}", fontsize=fontsize,
                    color=fontcolor)
            ax.text(x, y - 2, f"({round(x)}", fontsize=fontsize, color='purple')
    ax.scatter(x_vals, y_vals, color=color_vals)


def assert_target_dict(dict_to_check, allow_empty):
    if len(dict_to_check) > 0:
        sample = list(dict_to_check.items())[0]
        assert isinstance(sample[0], int) and isinstance(sample[1], Target)
    elif not allow_empty:
        raise Exception("Empty Target dict where there shouldnt be one")


def plot_tracks(ax, cones, subplot_title=None, x_label=None, y_label=None, show_plot=False, plot_fov_range=(50, 50),
                plot_fov=None, plot_view_angle=None, fontsize=6, fontcolor="red", plot_ids=True, car=None, ego=True):
    # validate input
    if isinstance(cones, TrackedCones):
        cone_dict = cones.cone_dict
    elif isinstance(cones, dict):
        assert_target_dict(cones, allow_empty=True)
        cone_dict = cones
    else:
        raise Exception(f"Invalid Value in plot_tracks: {type(cones)}: {cones} ")

    plot_cones_from_dict(ax, cone_dict, fontsize=fontsize, fontcolor=fontcolor, plot_ids=plot_ids)
    plot_car(ax, car=car, plot_fov=plot_fov, plot_view_angle=plot_view_angle, fov_range=plot_fov_range, ego=ego)
    print(f"Car fov: {np.degrees(car.fov)}")
    print(f"Car view angle: {np.tan(car._center_slope)}")
    set_axis_titles(ax, subplot_title, x_label, y_label)

    if show_plot:
        plt.show()


def visualize_detection(ax, cones, subplot_title="Detected cones", show_plot=True, car=None, plot_car=True,
                        x_title=None, y_title=None, first_col_colors=False):
    if first_col_colors:
        ax.scatter(np.round(cones[:, 1].astype(np.float32), 3), np.round(cones[:, 2].astype(np.float32), 3),
                   color=cones[:, 0])
    else:
        ax.scatter(np.round(cones[:, 0], 3), np.round(cones[:, 1], 3))

    ax.set_title(subplot_title)

    if x_title:
        ax.set(xlabel=x_title)
    if y_title:
        ax.set(ylabel=y_title)

    if plot_car and car:
        ax.scatter(car.x_global, car.y_global, c="green")

    if show_plot:
        plt.show()


def plot_covariance(ax, mean_x, mean_y, cov_mat: np.ndarray):
    assert cov_mat.shape == (2, 2)

    # from https://cookierobotics.com/007/#:~:text=2x2%20covariance%20matrix%20can%20be,followed%20by%20examples%20and%20explanations
    var_avg = (cov_mat[0, 0] + cov_mat[1, 1]) / 2
    b = cov_mat[0, 1]
    root_part = np.sqrt(((cov_mat[0, 0] - cov_mat[1, 1]) / 2) ** 2 + b ** 2)
    l1 = var_avg + root_part
    l2 = var_avg - root_part

    theta = None
    if b != 0:
        theta = np.arctan2(l1 - cov_mat[0, 0], b)
    elif cov_mat[0, 0] >= cov_mat[1, 1]:
        theta = 0
    elif cov_mat[0, 0] < cov_mat[1, 1]:
        theta = np.pi / 2
    else:
        raise Exception("Unreachable code reached")

    # draw parametric equation
    t = np.linspace(0, 2 * np.pi, 1000)
    l1, l2 = max(l1, 0), max(l2, 0)  # TODO: Got l1, l2 under
    x = np.sqrt(l1) * np.cos(theta) * np.cos(t) - np.sqrt(l2) * np.sin(theta) * np.sin(t)
    y = np.sqrt(l1) * np.sin(theta) * np.cos(t) + np.sqrt(l2) * np.cos(theta) * np.sin(t)

    ax.plot(x + mean_x, y + mean_y, c="black")


def plot_covariances(ax, points: List):
    if not points:
        return

    assert isinstance(points[0], Detection) or isinstance(points[0], Target)

    if isinstance(points[0], Detection):
        x_means = [det.det_x for det in points]
        y_means = [det.det_y for det in points]
        covariances = [det.cov_mat for det in points]
    elif isinstance(points[0], Target):
        x_means = [tar.state.x[0] for tar in points]
        y_means = [tar.state.x[1] for tar in points]
        covariances = [tar.state.P for tar in points]

    for i in range(len(covariances)):
        plot_covariance(ax, x_means[i], y_means[i], covariances[i])


def plot_dashed_line(ax, slope, intercept, a, b, color="orange"):
    x = np.linspace(a, b, 100000)
    y = slope * x + intercept
    ax.plot(x, y, linestyle='--', c=color)


class ConsecutiveFramePlotter:
    def __init__(self, tracker: MultiObjectTracker, car, def_ego: bool, x_label="Theta", y_label="Range", ground_truth=None,
                 default_range_to_plot_fov=(50, 50), fullscreen_default=False, predict_last=False):

        assert isinstance(car, Car)

        self.tracker = tracker  # Tracker object
        self.prev_car = None  # Car objects
        self.car = car
        self.x_lab = x_label
        self.y_lab = y_label
        self.prev_cones = {}  # TrackedCones
        self.curr_cones = {}  # TrackedCones
        self.prev_det = []  # List[Detections]
        self.curr_det = []  # List[Detections]
        self.prev_preds = None  # TrackedCones of prediction
        self.curr_preds = None  # TrackedCones
        self.execute_counter = 0  # times object ran execute
        self.ground_truth = ground_truth
        self.def_fov_range = default_range_to_plot_fov
        self.fullscreen_default = fullscreen_default
        self.predict_last = predict_last
        self.def_ego = def_ego
    def execute_and_plot(self, localization: Localization, detections: List[Detection], plot_type="pred", plot=True,
                         figure_title=None, covariances=False, plot_car=False, fov_plot_range=None,
                         fullscreen=None, ego=None) -> None:
        assert plot_type in {"pred", "det", "tracked"}  # Tracked is just tracked, det is tracked + detected, pred is + predicted
        """Plots new cones and keeps track of which to plot where"""
        self._copy_curr_values_to_prev_values()
        self.curr_det = copy.deepcopy(detections)
        self.car.move_car(localization)
        detections = self.car.detect(detections, filter_fov=True)  # What the car actually does detects within its fov.

        # Execute tracker
        if plot_type == "pred":
            self.tracker._predict(localization)
            self.curr_preds = copy.deepcopy(self.tracker.get_tracks())
            self.tracker.execute_without_predict_dont_use(localization, detections)
            self.curr_cones = self.tracker.get_tracks()
        else:
            self.curr_cones = self.tracker.execute(localization, detections)

        # Setting plotting params
        self.execute_counter += 1
        if not figure_title:
            figure_title = self._build_title(localization=localization)

        car = self.car if plot_car else None
        fov_range = fov_plot_range if fov_plot_range is not None else self.def_fov_range
        show_in_fullscreen = self.fullscreen_default if fullscreen is None else fullscreen
        ego = self.def_ego if not ego else ego

        # Plot timestep
        if plot:
            if plot_type == "pred":
                visualize_consec_frames_predictions_detections(self.prev_cones, self.curr_cones, self.prev_det,
                                                               self.curr_det, self.prev_preds, self.curr_preds,
                                                               "Global x-axis", "Global y-axis", fig_title=figure_title,
                                                               covariances=covariances, car=car,
                                                               plot_fov_range=fov_range,
                                                               prev_car=self.prev_car, fullscreen=show_in_fullscreen,
                                                               ground_truth=self.ground_truth, ego=ego)
            elif plot_type == "det":
                visualize_consec_frames_with_detections(self.prev_cones, self.curr_cones, self.prev_det,
                                                               self.curr_det,
                                                               "Global x-axis", "Global y-axis", fig_title=figure_title,
                                                               covariances=covariances, car=car,
                                                               plot_fov_range=fov_range,
                                                               prev_car=self.prev_car, fullscreen=show_in_fullscreen,
                                                               ground_truth=self.ground_truth)
            else:
                visualize_consec_frames(self.prev_cones, self.curr_cones, "Global x-axis", "Global y-axis")

    def _copy_curr_values_to_prev_values(self):
        self.prev_det = self.curr_det
        self.prev_car = copy.deepcopy(self.car)
        self.prev_cones = copy.deepcopy(self.curr_cones)
        self.prev_preds = copy.deepcopy(self.curr_preds)

    def _build_title(self, localization=None):
        if localization:
            ret_str = f"timestep {self.execute_counter}"
            for key, val in localization._asdict().items():
                if key not in {"qxx", "qyy"} and val != 0:
                    ret_str += f",{key}: {val}"
                elif key in {"qxx", "qyy"} and val != 1:
                    ret_str += f",{key}: {val}"
            return ret_str

        return f"timestep {self.execute_counter}"
