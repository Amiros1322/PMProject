"""
Create distance matricies between detections and predictions.
Consists of functions s.t.:

    Args: List[Detection] - list of new detections
          Dict[int, Target] - dictionary of targets after prediction

    Returns DistanceMessage object:
        distance_matrix: a matrix of integers that measures differences.

                         Shape=(# large matrix rows, # small matrix rows, number of values) where small matrix rows is the
                         number of predictions if # detections > # predictions. Otherwise it is the # of predictions.
                         Large matrix rows is the number of rows of the remaining matrix.

                         for some reason when not casted to integers the error
                         "TypeError: loop of ufunc does not support argument 0 of type numpy.float64 which has no
                                     callable sqrt method" appears - no idea why.

        Other needed outputs (that you can get with the helpers target_dict_to_np and detection_list_to_np):

        pred_matrix: np representation of targets. [id, color, **coordinates] in each row represents a target
        detection_matrix: np representation of detections [color, **coordinates] in each row
        ordDetIdx_to_detection: a dictionary mappping the ids of the detections in the input (not dist matrix) to the detections.
"""
from typing import Dict, NamedTuple, List
import numpy as np
from common import Localization, Detection, Track, Tracks, State, Target
from scipy.optimize import linear_sum_assignment

class DistanceMessage(NamedTuple):
    distance_matrix: np.ndarray
    pred_matrix: np.ndarray
    detection_mat: np.ndarray
    orgDetIdx_to_detection: dict

# TODO: IMPORTANT NOTE: for now we treat 'unknown' color as different from every color. Means that if a color measurement
# TODO: was inconclusive, it will be treated like a conclusive measurement of another color. Should change unless
# TODO: color detection is very reliable
def l2_norm(detections: List[Detection], predicted: Dict[int, Target], filter_color=True, diff_color_penalty=10**6) -> DistanceMessage:
    pred_mat = target_dict_to_np(predicted, no_color=False, no_z=False)  # [id, color, **coordinates] in each row
    det_mat, id_to_detection = detection_list_to_np(detections)  # [color, **coordinates] in each row

    no_id_pred_mat = _filter_unknown_colors_and_ids(pred_mat, 1, 4)

    # distance_mat[i, j] = distances betw prediction i and detection j in every dimension
    prediction_coordinates, detection_coordinates = no_id_pred_mat[:, np.newaxis, 1:].astype(np.float32), det_mat[:, 1:].astype(np.float32)
    distance_mat = prediction_coordinates - detection_coordinates
    assert distance_mat.shape == (pred_mat.shape[0], det_mat.shape[0], 3)  # TODO: Change to 2 if cam_z removed from detections

    # color_mat[i, j] = prediction i color == detection j color
    prediction_colors, detection_colors = no_id_pred_mat[:, np.newaxis, 0], det_mat[:, 0]
    color_mat = prediction_colors != detection_colors
    print(color_mat.shape)
    assert color_mat.shape == (pred_mat.shape[0], det_mat.shape[0])


    # distance_mat = distance_mat.astype(np.float32)
    distances = np.linalg.norm(distance_mat, axis=2)  # TODO: for some reason wont work with floats??

    distances = distances + color_mat * diff_color_penalty

    return DistanceMessage(distance_matrix=distances, pred_matrix=pred_mat, detection_mat=det_mat, orgDetIdx_to_detection=id_to_detection)


def _filter_unknown_colors_and_ids(prediction_mat: np.ndarray, color_col: int, unkown_value: int) -> np.ndarray:
    pred_mat = prediction_mat[:, 1:]  # remove ids
    pred_mat[:, color_col][pred_mat[:, color_col] == None] = unkown_value  # If a
    return pred_mat


def _dist_matrix_filter_color(distances, color_idx, inplace=False):
    """

    Args:
        distances: matrix of distances
        color_idx: the index where the color is stored in the matricies used for filtering

    Returns:

    """
    if not inplace:
        distances = distances.copy()

    inf_mask = distances[:, :, color_idx] == 0
    distances[inf_mask] = np.inf
    return distances


def target_dict_to_np(targets: Dict[int, Target], no_color=True, no_z=True):
    arr = [None]*len(targets)
    for idx, tar_item in enumerate(targets.items()):
        t_id, tar = tar_item
        if no_z and no_color:
            arr[idx] = ([t_id, tar.state.x[0], tar.state.x[1]])
        elif no_z and not no_color:
            arr[idx] = ([t_id, tar.cone_class, tar.state.x[0], tar.state.x[1]])
        else: # not noz
            if len(tar.state.x) == 3:
                tar_z = tar.state.x[2]
            else:
                tar_z = 0
            arr[idx] = ([t_id, tar.cone_class, tar.state.x[0], tar.state.x[1], tar_z])
    return np.array(arr, dtype=object)

def detection_list_to_np(detections: List[Detection]):
    arr = [None]*len(detections)
    id2_org_detection = dict()
    for idx, det in enumerate(detections):
        arr[idx] = [det.cone_class, det.cam_x, det.cam_y, det.cam_z]
        id2_org_detection[idx] = det
    return np.array(arr), id2_org_detection


def np_to_detection_list(arr: np.ndarray, first_color=True, const_cov=True, z_zero=True) -> List[Detection]:
    detections = []
    for row in arr:

        if z_zero:
            cam_z = 0
        else:
            cam_z = row[2 + first_color]

        cone_class = None
        if first_color:  # if color specified
            cone_class = row[0]

        cov_mat = None
        if const_cov:
            cov_mat = np.array([[1, 0], [0, 1]])

        det = Detection(cam_x=float(row[0 + first_color]), cam_y=float(row[1 + first_color]), cam_z=cam_z, cone_class=cone_class, cov_mat=cov_mat)

        detections.append(det)

    return detections