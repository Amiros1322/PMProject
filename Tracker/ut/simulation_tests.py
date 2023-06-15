import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from Tracker.src.TrackingUt import random_list_delete, simulated_cones_to_detection, global_to_ego_frame, ego_to_global_frame
from Tracker.src.parametric_curve_tracks import circle_track, bezier_track, straight_track, get_cone_locations
from Tracker.src.MultiObjectTracker import MultiObjectTracker
from Tracker.src.common import Localization, Car, SimulatedCone, Detection
from Tracker.src.simulated_data import first_frame, np_to_detection_list, add_noise, generate_track, detection_list_to_np
from Tracker.src.tracking_vis import ConsecutiveFramePlotter



def main():
    # t = straight_track(m=0, b=0)
    # cone_mat, ff_car, sim_cones = first_frame(t, points_as_cartesian=True)
    # ins_car, ins_sim_cones = initialize_simulation(t, get_cones_as_detections=False)

    # horizontal_track()
    diagonal_track()



"""
###########################################

Multi Object Tracker Tests

###########################################
"""

# Make List of detections to check this (static localization)

"""
Notes:
After update final corrected state covariance should be smaller, if detection noise is small.

Things to try:
Low and High levels of noise

See how covariance acts.

Test timestamp variance.

BUGS:
Car moving forward thinks that the next cones are the ones that were closest.
Survival update is weird

Ideas:
Difference between commits is 'initialize simulation' refactor. Try comparing output from first_frame.
It did not change much so you can use it from recent commit.

Maybe its because of the ego frame?
##############################
"""
def disappearing_static_cone(show_frame_by_frame=True):
    STATIC_LOCALIZATION = Localization(x_ego=0, y_ego=0, theta_ego=0, qyy=1, qxx=1)
    STANDARD_COVARIANCE = np.array([[1, 0], [0, 1]])

    # Getting the cone detection
    detections_polar, car, cone_objects = first_frame(straight_track(m=2, b=-4), car_t=0.1,
                                                                              samples=10, plot=False)

    tracker = MultiObjectTracker()
    plotter = ConsecutiveFramePlotter(tracker, car=car, x_label="Theta", y_label="Range", fullscreen_default=True)
    single_cone_detection = np_to_detection_list(detections_polar, first_color=True)[0]

    plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=False)  # First frame: no detections
    plotter.execute_and_plot(STATIC_LOCALIZATION, [single_cone_detection], plot=show_frame_by_frame)  # second frame: cone appears

    # add some noise to measurement in second and third frames
    for _ in range(2):
        noisy_cone = add_noise([single_cone_detection], noise_mu=0.5, noise_std=0.1, cov_noise_mu=0, cov_noise_std=0)
        plotter.execute_and_plot(STATIC_LOCALIZATION, noisy_cone, plot=show_frame_by_frame)

    # the cone disappears for 5 detections
    for _ in range(5):
        plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=show_frame_by_frame)

    # random cone locations
    for _ in range(10):
        magnitude = 1
        random_cone1 = Detection(cam_x=round(random.random() * magnitude, 3),
                                 cam_y=round(random.random() * magnitude, 3),
                                 cam_z=round(random.random() * magnitude, 3), cov_mat=STANDARD_COVARIANCE * 1.5, cone_class="blue")
        random_cone2 = Detection(cam_x=round(random.random() * magnitude, 3),
                                 cam_y=round(random.random() * magnitude, 3),
                                 cam_z=round(random.random() * magnitude, 3), cov_mat=STANDARD_COVARIANCE * 1.5, cone_class="yellow")
        plotter.execute_and_plot(STATIC_LOCALIZATION, [random_cone1, random_cone2], plot=show_frame_by_frame)

    # kill cones
    for _ in range(20):
        plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=show_frame_by_frame)
    print()


def full_track_cones(show_frame_by_frame=True, n_to_del=1, track=straight_track(m=2, b=-4), car_angle=0, fov=0.3 * np.pi,
                     samples=8, insert_noise=True, predict_last=False):
    STATIC_LOCALIZATION = Localization(x_ego=0, y_ego=0, theta_ego=0, qyy=1, qxx=1)
    STANDARD_COVARIANCE = np.array([[1, 0], [0, 1]])
    move_localization = Localization(x_ego=8, y_ego=0, theta_ego=np.radians(15), qyy=1, qxx=1)

    # Creating "ground truth"
    car, global_cones = initialize_simulation(track_func=track, car_angle=car_angle, fov=fov, samples=samples)
    cones = global_to_ego_frame(car, global_cones)

    # Initiating tracker and plotter
    tracker = MultiObjectTracker()
    plotter = ConsecutiveFramePlotter(tracker, car=car, x_label="Theta", y_label="Range", predict_last=predict_last)

    plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=False)  # First frame: no detections
    plotter.execute_and_plot(STATIC_LOCALIZATION, cones, plot=show_frame_by_frame, plot_car=True)  # second frame: cones appear

    for _ in range(7):
        new_det = copy.deepcopy(cones)
        random_list_delete(new_det, n=n_to_del)
        new_det = add_noise(new_det, noise_mu=1, noise_std=0.5, cov_noise_mu=0, cov_noise_std=0) if insert_noise else new_det
        plotter.execute_and_plot(move_localization, new_det, plot=show_frame_by_frame, plot_car=True)

    plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=show_frame_by_frame, plot_car=True)
    for _ in range(15):
        plotter.execute_and_plot(move_localization, [], plot=False)
    plotter.execute_and_plot(STATIC_LOCALIZATION, [], plot=show_frame_by_frame, plot_car=True)


def diagonal_track(predict_last=False):
    full_track_cones(predict_last=predict_last)


def horizontal_track():
    straight = straight_track(m=0, b=0)
    full_track_cones(show_frame_by_frame=True, n_to_del=2, track=straight, insert_noise=False)

"""
###########################################

Track generation tests

###########################################
"""


def test_cones():
    pass


def test_placement():
    _test_cone_placement([straight_track(m=3)], cone_dist=14)
    _test_cone_placement([circle_track(10)], cone_dist=6, samples=15, draw_pcurve=False)

    p0 = np.array([0, 0])
    p1 = np.array([10, 0])
    p2 = np.array([5, 10])
    p3 = np.array([15, 15])

    _test_cone_placement([bezier_track(p0, p1, p2, p3)], cone_dist=4, samples=10, draw_pcurve=True)


def _test_param_curve(curves_to_test):
    for curve in curves_to_test:
        x_lst = []
        y_lst = []
        for t in np.linspace(0, 1, 15):
            p1 = curve(t)
            x_lst.append(p1[0])
            y_lst.append(p1[1])

        plt.scatter(x_lst, y_lst)
        plt.show()


def _test_cone_placement(curves_to_test, cone_dist=4, samples=7, draw_pcurve=True, deriv_eps=0.02):
    for curve in curves_to_test:
        # left cones are blue, right cones yellow.  # TODO: In some cases (eg circular track) cones switch. Fix
        x_left_cones, x_right_cones, x_track = [], [], []
        y_left_cones, y_right_cones, y_track = [], [], []
        for t in np.linspace(deriv_eps, 1 - deriv_eps, samples):
            p1 = curve(t)
            x_track.append(p1[0])
            y_track.append(p1[1])

            # get cones
            cone1, cone2 = get_cones_at_t(t, curve, cone_distance=cone_dist, eps=deriv_eps)

            # Append cones to lists
            x_left_cones.append(cone1[0])
            y_left_cones.append(cone1[1])

            x_right_cones.append(cone2[0])
            y_right_cones.append(cone2[1])

        if draw_pcurve:  # draws parametric curve
            plt.scatter(x_track, y_track, c='red')
        plt.scatter(x_left_cones, y_left_cones, c='blue')
        plt.scatter(x_right_cones, y_right_cones, c='#8B8000')  # supposedly dark yellow
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()


def get_cones_at_t(t_approx, para_func, cone_distance=4, eps=0.02):
    p1 = para_func(min(t_approx + eps, 1))
    p2 = para_func(max(t_approx - eps, 0))

    # takes the normal of the line between p1, p2 and adds a cone distance away. p1 MUST have t greater than p2!!
    cone1, cone2 = get_cone_locations(p1, p2, dist_from_point=cone_distance)
    low_cone, high_cone = (cone1, cone2) if cone1[1] < cone2[1] else (cone2, cone1)

    if p1[0] > p2[0]:  # car goes 'right'
        # so left cone is on top
        blue_cone, yellow_cone = low_cone, high_cone
    else:  # car goes left and right cone is on top
        blue_cone, yellow_cone = high_cone, low_cone

    return blue_cone, yellow_cone


def place_cones(curves_to_test, cone_dist=4, samples=7, deriv_eps=0.02):
    for curve in curves_to_test:
        # left cones are blue, right cones yellow.  # TODO: In some cases (eg circular track) cones switch. Fix
        x_left_cones, x_right_cones, x_track = [], [], []
        y_left_cones, y_right_cones, y_track = [], [], []
        for t in np.linspace(deriv_eps, 1 - deriv_eps, samples):
            p1 = curve(t)
            x_track.append(p1[0])
            y_track.append(p1[1])

            # get cones
            cone1, cone2 = get_cones_at_t(t, curve, cone_distance=cone_dist, eps=deriv_eps)

            # Append cones to lists
            x_left_cones.append(cone1[0])
            y_left_cones.append(cone1[1])

            x_right_cones.append(cone2[0])
            y_right_cones.append(cone2[1])

def initialize_simulation(track_func, car_t=0.0, car_angle=0.35 * np.pi, fov=0.3 * np.pi, samples=8, cone_distance=4,
                          eps=0.02, get_cones_as_detections=True, min_max_normalize_cones=False) -> Tuple[Car, List[Detection]]:
    cones = generate_track(track_func, samples=samples, cone_distance=cone_distance, eps=eps, as_numpy=False)
    car = Car(track_func, car_t, eps, car_angle, fov)
    cones = simulated_cones_to_detection(cones) if get_cones_as_detections else cones
    return car, cones



if __name__ == "__main__":
    main()
