# Parametric curves for tracks
import numpy as np
import math


def bezier_track(p0, p1, p2, p3):
    def _inner_curve(t):  # TODO: Shouldnt this return two values?
        if type(t) == list:
            return [(1 - item)**3 * p0 + 3 * (1 - item)**2 * item * p1 + 3 * (1 - item) * item**2 * p2 + item**3 * p3 for item in t]
        elif type(t) == np.ndarray:
            point_lst = [(1 - item)**3 * p0 + 3 * ((1 - item)**2 * item * p1) + (3 * (1 - item) * item**2 * p2) + (item**3 * p3) for item in t]
            return  np.array([pnt[0] for pnt in point_lst]), np.array([pnt[1] for pnt in point_lst])
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return _inner_curve

def straight_track(m=0, b=0, length=100):
    in_len = length - 1
    def _inner_straight(t):
        return in_len * t, m * (in_len * t) + b

    return _inner_straight


def circle_track(radius, center_x=0, center_y=0):
    def _inner_circle(t):
        t = t * 2*np.pi
        x = radius * np.cos(t) + center_x
        y = radius * np.sin(t) + center_y
        return x, y

    return _inner_circle


def get_cone_locations(point1, point2, dist_from_point=15, eps=0.0001):
    # find slope of line between points
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])

    # find slope of normal line
    slope_normal = -1 / (slope + eps)

    # the midpoint between the two points
    line_x, line_y = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2

    # we want to find the points that are dist_from_point away from the line.
    # using euclidean distance and the normal's equation y = slope_normal * (x - line_x) + line_y:
    intercept = -slope_normal*line_x + line_y
    a = slope_normal ** 2 + 1
    b = -2 * line_x + 2*slope_normal*intercept - 2*line_y*slope_normal
    c = line_x**2 + line_y**2 + intercept**2 - dist_from_point**2 - 2*line_y*intercept

    x_sol1, x_sol2 = quadratic_equation(a, b, c)
    y_sol1, y_sol2 = slope_normal * x_sol1 + intercept, slope_normal * x_sol2 + intercept

    return (x_sol1, y_sol1), (x_sol2, y_sol2)


def quadratic_equation(a, b, c):
    root = math.sqrt(b ** 2 - 4 * a * c)
    return (-b + root) / (2 * a), (-b - root) / (2 * a)