import random
from common import SimulatedCone
from typing import List
import numpy as np
import math

"""
Input: detections of two opposite cones on a straight section of the track.

Output: the distance between the camera and the part of the track between the cones
"""


def two2three_d(cone_left, cone_right, focal_length, image_length, track_width=3, cone_width=4):
    # cones <- [x, y, width, height]
    delta_phys = track_width + cone_width  # real physical distance between cones (by track regulations)
    delta_px = cone_right[0] - cone_left[0]  # dist between cone centers in pixels on image plane
    two_cone_angle = 2 * math.atan(delta_phys / 2 * focal_length)

    # Camera may not be in the center of the track.
    # So we can't just take two_cone_angle/2 - need point in front of camera. On the line between the cones.

    t = ((image_length / 2) - cone_left[0]) / delta_px  # Where betw. cones the middle of the picture is.

    assert 0 <= t <= 1

    # The interpolated pnt t is also the point on line betw. cones in front of camera in phys world.
    left_length, right_length = t * delta_phys, (1 - t) * delta_phys
    left_angle = 2 * math.atan(left_length / 2 * focal_length)
    right_angle = 2 * math.atan(right_length / 2 * focal_length)

    print(f"Angle Error (radians): {left_angle + right_angle - two_cone_angle}")

    # finally, calculate the distances we want
    left_dist = left_length / math.tan(left_angle)
    right_dist = right_length / math.tan(right_angle)

    print(f"Left distance: {left_dist}, Right distance: {right_dist}. \nError: {left_dist - right_dist}")
    return (left_dist + right_dist) / 2


def cart_to_polar(points: np.ndarray, car_x, car_y, car_slope, inplace=True):
    """
    Input: a numpy array of points in cartesian coordinates, the car coordinates
    Output: A numpy array of points in polar coordinates

    Takes distances with sqrt( (x_point - x_car)**2 + (y_point - y_car)**2). Each op vectorized
    """
    if not inplace:
        points = points.copy()

    # take difference of point and car coordinates - used for both distance and angle calculation
    points[:, 0] = points[:, 0] - car_x
    points[:, 1] = points[:, 1] - car_y

    # Size of each row i in points is now the size of the line between point i and the car
    distances = np.linalg.norm(points, axis=1)

    # Direction vector of the car's angle
    car_direction = np.array([1, car_slope])

    # Needed for angle calculation
    dot_product = np.dot(car_direction, points.T)

    # Calculate the dot product and norms of v1 and v2
    norm_car = np.linalg.norm(car_direction)

    # Calculate the angle between l1 and the line between each point and (x1, y1)
    angles = np.arccos(dot_product / (norm_car * distances))

    ret_pnts = np.array([angles, distances]).T
    return ret_pnts


def random_dict_delete(d: dict, n: int):
    """
    Deletes n random elements from the dictionary

    """
    for _ in range(n):
        idx_to_del = random.choice(list(d.keys()))
        del d[idx_to_del]


def random_list_delete(l: list, n: int):
    """
    Deletes n random elements from the list

    """
    for _ in range(n):
        idx_to_del = random.choice(list(range(len(l))))
        del l[idx_to_del]


def simulated_cones_to_numpy(cones: List[SimulatedCone]) -> np.ndarray:
    return np.array([[cone.x, cone.y] for cone in cones])
