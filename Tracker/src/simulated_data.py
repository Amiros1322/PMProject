import random
from typing import List, NamedTuple, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
from Tracker.src.parametric_curve_tracks import straight_track, circle_track, get_cone_locations, bezier_track
from Tracker.src.TrackingUt import random_dict_delete, cart_to_polar, simulated_cones_to_numpy, simulated_cones_to_detection
from Tracker.src.MultiObjectTracker import MultiObjectTracker
from Tracker.src.common import Localization, Detection, Car, SimulatedCone
from Tracker.src.distance_functions import target_dict_to_np, detection_list_to_np, np_to_detection_list
from Tracker.src.tracking_vis import visualize_detection, visualize_labeling, visualize_ground_truth

class Frame(NamedTuple):
    time: float

    # At time t, these are the car's coordinates relative to the car's position at t-1.
    x_ego: float
    y_ego: float
    theta_ego: float

    detections: np.ndarray  # detections in camera reference frame?? No. Those are 2 different things.


class Params:
    dt: float  # for car movement?
    n_frames: int
    route_func: type(lambda x: x)  # function that returns a route after a specific frame.
    sight_depth: float
    FOV: float  # angle from car that cones are detected in
    move_std: float
    meas_std: float


class Tracks:
    id: int
    p_alive: float


# TODO: Erease if object works. This is for debugging
class CarStruct:
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


def create_simulated_data2(params: Params):
    # Generate track, first frame, etc.
    frame1, detections_polar, car, real_cones, cartesian_points = first_frame(straight_track(m=2, b=-4), car_t=0.1, samples=10)
    cart_as_list = np_to_detection_list(cartesian_points, first_color=False)

    # create tracker and populate with inital detections
    tracker = MultiObjectTracker(default_alpha=0.95, alpha_nonexist_thresh=0.5)
    tracked_cones = tracker.execute(Localization(x_ego=0, y_ego=0, theta_ego=0, qxx=1, qyy=1), cart_as_list, dtime=params.dt)

    # do weird stuff to the points
    np.random.shuffle(cartesian_points)
    modified_cartesian = delete_random_rows(cartesian_points, 4)
    modified_cartesian = np_to_detection_list(modified_cartesian, first_color=False)

    for _ in range(20):
        unused_tracks = tracker.execute(Localization(x_ego=0, y_ego=0, theta_ego=0, qxx=1, qyy=1), modified_cartesian,
                                        dtime=params.dt)
        np.random.shuffle(unused_tracks)

    # track same points after
    tracked_two = tracker.execute(Localization(x_ego=0, y_ego=0, theta_ego=0, qxx=1, qyy=1), modified_cartesian,
                                  dtime=params.dt)

    # visualize step
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    visualize_ground_truth(ax[0, 0], real_cones, car=car, plot_view_angle=True, show_plot=False)
    visualize_detection(ax[0, 1], detections_polar, subplot_title="Detections in polar coordinates", show_plot=False)
    visualize_labeling(ax[1, 0], target_dict_to_np(tracked_cones.cone_dict), subplot_title="Old ids")
    visualize_labeling(ax[1, 1], target_dict_to_np(tracked_two.cone_dict), subplot_title="New ids")

    plt.show()


def initialize_car(track_func, car_t: float, car_angle: float, fov: float, eps=0.02):
    car = CarStruct()
    p_start = track_func(car_t)
    p_car = track_func(car_t + eps)
    car_slope = (p_start[1] - p_car[1]) / (p_start[0] - p_car[0] + 0.000001)
    car.x_global, car.y_global, car.theta_global, car.fov = p_car[0], p_car[1], car_angle, fov
    return car


def create_simulated_data(params: Params) -> List[Frame]:
    # creates simulated track, simulated car, and the first detections (polar coordinates)
    frame1, detections, car, real_cones, cartesian_points = first_frame(straight_track(m=2, b=-4), car_t=0.1,
                                                                        samples=10)
    frames = [frame1]

    # det_camera = polar_to_camera(detections)

    # First assignment of ids. col1 = id, col > i is a detection variable
    assignment = assign_ids(detections)
    existing_tracks = assignment[:, 0].tolist()  # start tracking cones
    init_p_exist = 0.5

    predictions = predict(assignment, car, params.dt)

    # shuffles predictions to test association
    predictions_old = predictions.copy()
    np.random.shuffle(predictions)
    random_dict_delete(predictions, 2)

    new_ass = associate(detections, predictions, existing_tracks)

    # visualize step
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    visualize_ground_truth(ax[0, 0], real_cones, car=car, plot_view_angle=True, show_plot=False)
    visualize_detection(ax[0, 1], detections, subplot_title="Detections in polar coordinates", show_plot=False)
    visualize_labeling(ax[1, 0], predictions_old, subplot_title="Old ids (polar coordinates)")
    visualize_labeling(ax[1, 1], new_ass, subplot_title="New ids (polar coordinates)")

    plt.show()

    # noise for measurements and car movement
    meas_noise = np.random.normal(0, params.meas_std, size=(2, params.n_frames))
    move_noise = np.random.normal(0, params.move_std, size=(3, params.n_frames))


# detections euclidean?
def predict(detections, car, dtime, qxx=1, qyy=1):
    F = np.array([[np.cos(car.theta_ego), -np.sin(car.theta_ego)],
                  [np.sin(car.theta_ego), np.cos(car.theta_ego)]])
    u = np.array([car.x_ego, car.y_ego])
    Q = dtime * np.array([[qxx ** 2, 0], [0, qyy ** 2]])

    P = np.array([[1, 0], [0, 1]])

    # cone predictions
    predictions = {i: {'cov': None, 'pred': None} for i in detections[:, 0]}
    for row in detections:  # for every cone
        predictions[row[0]]['pred'] = F @ row[1:] + u
        predictions[row[0]]['cov'] = F @ P @ F.T + Q

    # car prediction
    return predictions


"""
Stores cone points in a matrix. If # new detections > predictions,

Keep the matrix. on most frames the association will not change.
Once a change happens, then create a new assoc mat. Otherwise do everything inplace.  
"""


# TODO: Add a score threshhold where you don't add a cone if it crosses the threshold
def associate(detections, predictions, existing_tracks):
    # currently using l2 norm
    pred_vals = np.array([pred['pred'] for pred in predictions.values()])
    id2org_pred = {row: predictions[row]['pred'] for row in predictions}
    det_idx2det_value = {i: detections[i, :] for i in np.arange(detections.shape[0])}
    if detections.shape[0] > pred_vals.shape[0]:  # More detected than were predicted
        diff_vect = detections[:, np.newaxis, :] - pred_vals
        distances = np.linalg.norm(diff_vect, axis=2)

        row_ind, col_ind = linear_sum_assignment(distances)

        org_pred2new_pred = {tuple(pred_vals[col_ind[i]]): tuple(detections[row_ind[i]]) for i in
                             np.arange(len(col_ind))}

        ordered_detections = np.array([org_pred2new_pred[tuple(id2org_pred[id])] for id in predictions[:, 0]])

        new_ass = np.concatenate([predictions[:, 0].reshape(-1, 1), ordered_detections], axis=1)

        max_id = max(existing_tracks)
        # add new ids to detected cones
        for detection in detections:
            if tuple(detection) not in org_pred2new_pred.values():
                # add new row to assigned detections
                assigned_row = np.concatenate([np.array([max_id + 1]).reshape(1), detection])
                new_ass = np.concatenate([assigned_row.reshape(1, -1), new_ass], axis=0)

                # update tracks
                max_id += 1
                existing_tracks.append(max_id)  # carries over to outside scope

    else:  # more predicted or equal
        distances = np.linalg.norm(pred_vals[:, np.newaxis, :] - detections, axis=2)
        row_ind, col_ind = linear_sum_assignment(distances)
        org_pred2new_pred = {tuple(pred_vals[row_ind[i]]): tuple(detections[col_ind[i]]) for i in
                             np.arange(len(row_ind))}
        ordered_detections = np.array([org_pred2new_pred[tuple(id2org_pred[id])] for id in predictions[:, 0]])
        new_ass = np.concatenate([predictions[:, 0].reshape(-1, 1), ordered_detections], axis=1)

    return new_ass


"""
First assignment of ids to cones
"""


def assign_ids(detections):
    ids = np.arange(detections.shape[0]).reshape(-1, 1)
    assignment = np.concatenate([ids, detections], axis=1)
    return assignment


"""
gets detections out of points that it is possible for the car to detect.
For now it is the identity function
"""


def detect(detectable_points):
    return detectable_points


def add_noise(detections: List[Detection], noise_mu: float, noise_std: float, cov_noise_mu: float,
              cov_noise_std: float) -> List[Detection]:
    if len(detections) == 0:
        return []
    params = dynamically_get_detection_params(detections[0])
    noise_matrix = np.random.normal(noise_mu, noise_std, size=(len(detections), len(params)))
    covariance_noises = np.random.normal(cov_noise_mu, cov_noise_std, size=(len(detections), 2))
    noisy_detections = _add_noise_to_detection_params(detections, params, noise_matrix, covariance_noises)
    return noisy_detections


def _add_noise_to_detection_params(detections: List[Detection], params: List[str], noise_matrix: np.ndarray,
                                   cov_noises: np.ndarray) -> List[Detection]:
    new_detections = [None] * len(detections)
    for i in range(len(detections)):
        curr_detection = detections[i]

        # add noise to covariances
        cov1 = curr_detection.cov_mat[0, 1] + cov_noises[i][0]
        cov2 = curr_detection.cov_mat[1, 0] + cov_noises[i][1]
        new_cov = np.array([[1, cov1], [cov2, 1]])

        # Add noise to detections
        det_dict = curr_detection._asdict()
        noisy_detection_params = np.array([det_dict[key] for key in params]) + noise_matrix[i, :]

        # create new detection
        new_detections[i] = Detection(**{params[i]: noisy_detection_params[i] for i in range(len(params))},
                                      cone_class=curr_detection.cone_class, cov_mat=new_cov)
    return new_detections


def get_possible_covariance_values_square_matrix(matrix: np.ndarray) -> int:
    return matrix.shape[0] ** 2 - matrix.shape[0]


def dynamically_get_detection_params(detection: Detection) -> List[str]:
    """
    Get all parameters from Detection other than 'cov_mat' and 'is_blue' during runtime, no matter what they are

    Args:
        detection: Detection object

    Returns: all params except for cov_matrix and is_blue

    """
    params = [key for key in detection._asdict().keys() if key not in ["cov_matrix", "cone_class", "cov_mat"]]
    if len(params) > 4:
        raise Exception("Strong Warning - too many parameters to add noise to in Detection object."
                        "Are you sure it was set correctly?")
    return params


def initialize_simulation(track_func, car_t=0.0, car_angle=0.35 * np.pi, fov=0.3 * np.pi, samples=8, cone_distance=4,
                          eps=0.02) -> Tuple[np.ndarray, Car, List[SimulatedCone]]:
    cones = generate_track(track_func, samples=samples, cone_distance=cone_distance, eps=eps, as_numpy=False)
    car = Car(track_func, car_t, eps, car_angle, fov)
    cones_as_detections = simulated_cones_to_detection(cones)
    cones_as_np, _ = detection_list_to_np(cones_as_detections)
    return cones_as_np, car, cones


def first_frame(track_func, car_t=0.0, car_angle=0.35 * np.pi, fov=0.3 * np.pi, samples=8, cone_distance=4, eps=0.02,
                points_as_cartesian=False, plot=False) -> Tuple[np.ndarray, Car, List[SimulatedCone]]:
    # Parameters for affecting cone placment: samples, cone_dist [from para curve]
    cones = generate_track(track_func, samples=samples, cone_distance=cone_distance, eps=eps, as_numpy=False)
    car = Car(track_func, car_t, eps, car_angle, fov)
    det_cones = simulated_cones_to_detection(cones)  # TODO: Possible bug here
    sim_cones = car.filter_out_of_fov(det_cones)
    sim_cones = car.detect(sim_cones)  # further filtering based on cones car can actually detect. Currently id function

    plot_cones, all_cones = simulated_cones_to_numpy(sim_cones), simulated_cones_to_numpy(cones)

    # transformation from the 2d global xy coordinates to polar coordinates
    polar_points = cart_to_polar(plot_cones, car.x_global, car.y_global, car._center_slope)

    # Add polar coords to simulated cone objects, and add color to polar arrays
    for i, polar_point in enumerate(polar_points):
        sim_cones[i].angle = polar_point[0]
        sim_cones[i].range = polar_point[1]

    cart_coords = len(plot_cones[0]) if plot_cones.size != 0 else 2
    color_col = np.array([cone.cone_class for cone in sim_cones])
    polar_points = np.concatenate((color_col.reshape(len(sim_cones), 1), polar_points.reshape(len(polar_points), 2)),
                                  axis=1)
    cart_points = np.concatenate(
        (color_col.reshape(len(sim_cones), 1), plot_cones.reshape(len(plot_cones), cart_coords)), axis=1)

    points_to_ret = cart_points if points_as_cartesian else polar_points
    # visualize cones
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=False, sharey=False)
        visualize_ground_truth(axs[0], plot_cones, car=car, show_plot=False, subplot_title="Cart ground truth",
                               first_column_color=False, plot_view_angle=True)
        visualize_detection(axs[1], points_to_ret, show_plot=False, car=car, plot_car=False, first_col_colors=True)
        plt.show()

    return points_to_ret, car, sim_cones


def generate_track(track_func, samples=8, cone_distance=4, eps=0.02, num_orange_cones=0, as_numpy=False) -> List[SimulatedCone]:
    """
    A function to generate the 'ground truth' of the simulation. Takes track parameters and returns cones.

    Args:
        track_func: (function) Parametric function of the track. track_func(t) = (x, y) at t
        samples: (int) number of cone pairs to create
        cone_distance: (float) distance between cones on either side of the track
        eps: (float) two t values are taken to estimate track_func slope at sampled points. eps is the distance between them
        num_orange_cones: (int) number of orange cones at the start of the track

    Returns:
        A list of simulated cone structs with x, y, (integers) and color (string) properties

    """
    assert type(samples) == int and type(samples) == int
    # Sample points along the track curve
    t_vals = np.linspace(0, 1, samples)
    t_low = t_vals - eps
    t_top = t_vals + eps

    t_low[t_low < 0] = 0
    t_top[t_top > 1] = 1

    p_low, p_top = track_func(t_low), track_func(t_top)

    # Create ground truth cones
    real_cones = [None] * (len(t_vals) * 2)
    for i in range(len(t_vals)):
        # find cones off of track curve points
        low_pnt, high_pnt = (p_low[0][i], p_low[1][i]), (p_top[0][i], p_top[1][i])
        cone1, cone2 = get_cone_locations(low_pnt, high_pnt, dist_from_point=cone_distance)
        sim_cone1, sim_cone2 = get_simulated_cone_pair(cone1, cone2, low_pnt, high_pnt)

        if as_numpy:
            real_cones[i], real_cones[len(t_vals) + i] = cone1, cone2
        else:
            real_cones[i], real_cones[len(t_vals) + i] = sim_cone1, sim_cone2

    # Note: real cones is not assumed to be ordered
    # real_cones = np.array(real_cones)
    # real_cones.reshape(-1, 2)

    assert None not in real_cones

    return real_cones


def get_simulated_cone_pair(cone1: tuple, cone2: tuple, low_t_point: tuple, high_t_point: tuple) -> List[SimulatedCone]:
    higher_color, lower_color = None, None
    if low_t_point[0] < high_t_point[0]:  # car moving 'towards the right'
        higher_color = "blue"
        lower_color = "yellow"
    elif low_t_point[0] > high_t_point[0]:  # car moving 'towards the left'
        higher_color = "yellow"
        lower_color = "blue"
    else:  # car moving straight up or straight down

        if low_t_point[1] > high_t_point[1]:  # moving down
            left_color = "yellow"
            right_color = "blue"
        else:  # moving up
            left_color = "blue"
            right_color = "yellow"

        left_cone, right_cone = (cone1, cone2) if cone1[0] < cone2[0] else (cone2, cone1)
        return [SimulatedCone(x=left_cone[0], y=left_cone[1], cone_class=left_color, range=None, angle=None),
                SimulatedCone(x=right_cone[0], y=right_cone[1], cone_class=right_color, range=None, angle=None)]

    higher_cone, lower_cone = (cone1, cone2) if cone1[1] > cone2[1] else (cone2, cone1)
    return [SimulatedCone(x=higher_cone[0], y=higher_cone[1], cone_class=higher_color, range=None, angle=None),
            SimulatedCone(x=lower_cone[0], y=lower_cone[1], cone_class=lower_color, range=None, angle=None)]


"""
Input: car's global x, y, values, it's angle relative to global x, its fov (ANGLES IN RADIANS)
Returns slope, intercept of the two lines that mark the border of the car's fov
"""


def get_fov_lines(x, y, fov, view_angle):
    # Calculate the half angle of the field of view
    half_fov = fov / 2

    # Calculate the angles of the two lines that form the border of the field of view
    left_angle = view_angle - half_fov
    right_angle = view_angle + half_fov

    # Calculate the slope of the center line
    center_slope = np.tan(view_angle)

    # Calculate the slopes of the two lines relative to the center line
    left_relative_slope = np.tan(left_angle - view_angle)
    right_relative_slope = np.tan(right_angle - view_angle)

    # Calculate the y-intercept of each line
    left_intercept = y - (left_relative_slope + center_slope) * x
    right_intercept = y - (right_relative_slope + center_slope) * x
    center_intercept = y - center_slope * x

    # Return the equations of the two lines
    return center_slope, center_intercept, left_relative_slope, left_intercept, left_angle, right_relative_slope, right_intercept, right_angle


def polar_to_camera(polar_coordinates):
    """
    Gets the coordinates of the points in camera coordinates.

    Note that as we are in 2d for the simulated data, a camera frame detection
    is [x_cam, distance] instead of [x_cam, y_cam, distance]
    """
    assert len(polar_coordinates.shape) == 2  # making sure we get [angle, distance] as input

    x_cam = polar_coordinates[:, 1] * np.sin(polar_coordinates[:, 0])
    cam_coords = np.array([x_cam, polar_coordinates[:, 1]]).T

    return cam_coords


"""
Input: characteristics of the fov lines, matrix of x and y of points
Output: Matrix of detectable points
"""


# TODO: Will take points in back quadrant too. Make it so it doesnt.
def get_points_between_lines(car, cone_points: List[SimulatedCone]):
    assert isinstance(cone_points[0], SimulatedCone)
    # Convert the points to a NumPy array
    points = np.array(cone_points)[:, :2].astype(np.float32)
    cone_points = np.array(cone_points)

    left_slope = car.left_rel_slope + car.center_slope
    right_slope = car.right_rel_slope + car.center_slope

    # Calculate the y values for each line at the x coordinates of the points
    cone_x_vals = points[:, 0]
    y_left = left_slope * cone_x_vals + car.left_intercept
    y_right = right_slope * cone_x_vals + car.right_intercept

    # Find the points where y is between y_left and y_right
    cone_y_vals = points[:, 1]
    mask = np.logical_and(cone_y_vals >= np.minimum(y_left, y_right), cone_y_vals <= np.maximum(y_left, y_right))

    # Return the points that satisfy the condition
    cone_points = cone_points[mask]
    seen_points = [SimulatedCone(x=row[0], y=row[1], cone_class=row[2]) for row in cone_points]

    return seen_points


def get_points_between_lines_numpy(car, points):
    # Convert the points to a NumPy array
    points = np.array(points)
    left_slope = car.left_rel_slope + car.center_slope
    right_slope = car.right_rel_slope + car.center_slope
    # Calculate the y values for each line at the x coordinates of the points
    y_left = left_slope * points[:, 0] + car.left_intercept
    y_right = right_slope * points[:, 0] + car.right_intercept

    # Find the points where y is between y_left and y_right
    mask = np.logical_and(points[:, 1] >= np.minimum(y_left, y_right), points[:, 1] <= np.maximum(y_left, y_right))

    # Return the points that satisfy the condition
    return points[mask]


def delete_random_rows(arr, num_to_del) -> np.ndarray:
    for _ in range(num_to_del):
        arr = np.delete(arr, random.randint(0, arr.shape[0] - 1), axis=0)
    return arr


# test_placement()
p = Params()
p.meas_std = 0.1
p.move_std = 0.2
p.dt = 0.01
p.n_frames = 10
# create_simulated_data2(p)
