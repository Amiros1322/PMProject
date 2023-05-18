import matplotlib.pyplot as plt
from common import Detection, Target
from typing import Dict, NamedTuple, List  # TODO: Ask about dataclasses
from distance_functions import target_dict_to_np
from MultiObjectTracker import TrackedCones, MultiObjectTracker
from common import Localization,Detection
import numpy as np
import copy

def visualize_consec_frames(prev_cones, curr_cones, x_label, y_label, fontsize=6, color="red", colored_cones=True):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.5)
    visualize_labeling(ax[0], prev_cones, subplot_title="Previous frame cones", x_label=x_label, y_label=y_label, fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    visualize_labeling(ax[1], curr_cones, subplot_title="Current frame cones", x_label=x_label, y_label=y_label, fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    plt.show()



def visualize_consec_frames_with_detections(prev_cones, curr_cones, x_label, y_label, fontsize=6, color="red", colored_cones=True):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.5)
    visualize_labeling(ax[0, 0], prev_cones, subplot_title="Previous tracked cones", x_label=x_label, y_label=y_label, fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    visualize_labeling(ax[0, 1], curr_cones, subplot_title="Current tracked cones", x_label=x_label, y_label=y_label, fontsize=fontsize, fontcolor=color, colored_cones=colored_cones)
    plt.show()


def visualize_ground_truth(ax, cones, car=None, subplot_title="Ground Truth", show_plot=True, plot_car=True,
                           plot_view_angle=False, first_column_color=False):
    x_idx, y_idx, color_idx = (1, 2, 0) if first_column_color else (0, 1, None)
    start_draw = -10
    if car:
        if plot_car:
            ax.scatter(car.x_global, car.y_global, c="green")

        if plot_view_angle:
            plot_dashed_line(ax, car._center_slope, car._center_intercept, start_draw,
                             cones[:, x_idx].max(), color="blue")

        plot_dashed_line(ax, car._center_slope + car._left_rel_slope, car._left_intercept, start_draw, cones[:, x_idx].max())
        plot_dashed_line(ax, car._center_slope + car._right_rel_slope, car._right_intercept, start_draw, 100,
                         color="purple")

    if first_column_color:
        ax.scatter(cones[:, x_idx], cones[:, y_idx], color=cones[:, color_idx])
    else:
        ax.scatter(cones[:, x_idx], cones[:, y_idx])

    ax.set_title(subplot_title)
    if show_plot:
        plt.show()


def visualize_labeling(ax, labeled_detections, subplot_title=None, x_label=None, y_label=None, show_plot=False,
                       car=None, colored_cones=False, fontsize=6, fontcolor="red"):
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
    if isinstance(labeled_detections, Dict):
        labeled_detections = target_dict_to_np(labeled_detections, no_color=not colored_cones)
    elif isinstance(labeled_detections, TrackedCones):
        labeled_detections = target_dict_to_np(labeled_detections.cone_dict, no_color=not colored_cones)


    if labeled_detections.size == 0:
        ax.scatter([], [])
    else:
        if colored_cones:
            ax.scatter(labeled_detections[:, -2].astype(float).tolist(), labeled_detections[:, -1].astype(float).tolist(), color=labeled_detections[:, -3].tolist())
        else:
            assert labeled_detections.dtype in {np.float64, np.float32, np.float16}
            ax.scatter(labeled_detections[:, -2], labeled_detections[:, -1])

    if car:
        ax.scatter(car.x_global, car.y_global, c="green")

    if colored_cones:
        for lab_id, _, x, y in labeled_detections:
            ax.text(x, y, f"id:{lab_id}", fontsize=fontsize, color=fontcolor)
    else:
        for lab_id, x, y in labeled_detections:
            ax.text(x, y, f"id:{lab_id}", fontsize=fontsize, color=fontcolor)

    if subplot_title:
        ax.set_title(subplot_title)

    if x_label and y_label:
        ax.set(xlabel=x_label, ylabel=y_label)
    elif x_label:
        ax.set(x_label=x_label)
    elif y_label:
        ax.set(ylabel=y_label)

    if show_plot:
        plt.show()


def visualize_detection(ax, cones, subplot_title="Detected cones", show_plot=True, car=None, plot_car=True,
                        x_title=None, y_title=None, first_col_colors=False):
    if first_col_colors:
        ax.scatter(np.round(cones[:, 1].astype(np.float32), 3), np.round(cones[:, 2].astype(np.float32), 3), color=cones[:, 0])
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


def plot_dashed_line(ax, slope, intercept, a, b, color="orange"):
    x = np.linspace(a, b, 100000)
    y = slope * x + intercept
    ax.plot(x, y, linestyle='--', c=color)


class ConsecutiveFramePlotter:
    def __init__(self, tracker: MultiObjectTracker, x_label="Theta", y_label="Range"):
        self.x_lab = x_label
        self.y_lab = y_label
        self.prev_cones = {}  # TrackedCones
        self.curr_cones = {}  # TrackedCones
        self.tracker = tracker

    def execute_and_plot(self, localization: Localization, detections: List[Detection], plot=True) -> None:
        """Plots new cones and keeps track of which to plot where"""
        self.prev_cones = copy.deepcopy(self.curr_cones)
        self.curr_cones = self.tracker.execute(localization, detections)

        if plot:
            visualize_consec_frames(self.prev_cones, self.curr_cones, "Range", "Theta")
