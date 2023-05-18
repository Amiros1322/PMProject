import math
import time
"""
Input: detections of two opposite cones on a straight section of the track.

Output: the distance between the camera and the part of the track between the cones
"""
def two2three_d(cone_left, cone_right, focal_length, image_width, track_width=4, cone_width=0.228):
    # Without loss of generality:
    if len(cone_left.shape) > 1:  # take out of nested tensor
        cone_left, cone_right = cone_left[0], cone_right[0]
    if cone_left[0] > cone_right[0]:
        cone_left, cone_right = cone_right, cone_left

    # cones <- [x, y, width, height]
    delta_phys = track_width + cone_width  # real physical distance between cones (by track regulations)
    delta_px = cone_right[0] - cone_left[0]  # dist between cone centers in pixels on image plane
    two_cone_angle = 2*math.atan(delta_phys / 2*focal_length)

    # Camera may not be in the center of the track.
    # So we can't just take two_cone_angle/2 - need point in front of camera. On the line between the cones.

    t = ((image_width/2) - cone_left[0]) / delta_px  # Where betw. cones the middle of the picture is.

    try:
        assert 0 <= t <= 1
    except AssertionError:
        print("INVALID T VALUE")
        print(f"CONE_L: {cone_left}, CONE_R:{cone_right}")
        print(t)
        time.sleep(1)


    # The interpolated pnt t is also the point on line betw. cones in front of camera in phys world.
    left_length, right_length = t*delta_phys, (1-t)*delta_phys
    left_angle = 2*math.atan(left_length / 2*focal_length)
    right_angle = 2*math.atan(right_length / 2*focal_length)

    print(f"Left Angle: {left_angle}, right angle: {right_angle}\nAngle Error (radians): {left_angle + right_angle - two_cone_angle}")

    # finally, calculate the distances we want
    left_dist = left_length / math.tan(left_angle)
    right_dist = right_length / math.tan(right_angle)

    print(f"Left distance: {left_dist}, Right distance: {right_dist}. \nError: {left_dist - right_dist}")
    return (left_dist + right_dist) / 2
