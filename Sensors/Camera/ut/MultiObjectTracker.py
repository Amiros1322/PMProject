import random
from typing import Dict, NamedTuple, List
import numpy as np
from common import Localization, Detection, Track, Tracks, get_detection_state
from distance_functions import l2_norm, target_dict_to_np, detection_list_to_np
from scipy.optimize import linear_sum_assignment
from TrackingUt import cart_to_polar

class State(NamedTuple):
    x: np.ndarray
    P: np.ndarray


class Target(NamedTuple):
    id: int
    cone_class: str
    existence_probability: float
    state: State


class TrackedCones(NamedTuple):
    cone_dict: Dict[id, Target]


def add_new_target_to_dict_at_tarid(tar_id: int, tar_class: str, tar_state_arr: np.ndarray, tar_cov_mat: np.ndarray,
                           tar_exist_prob: float, dict_to_update: Dict):
    new_state = State(x=tar_state_arr, P=tar_cov_mat)
    new_target = Target(id=tar_id, cone_class=tar_class, state=new_state,
                        existence_probability=tar_exist_prob)
    dict_to_update[tar_id] = new_target


class MultiObjectTracker:
    _distance_functions = {"l2_norm": l2_norm, "l2norm": l2_norm}
    def __init__(self, default_alpha=0.95, alpha_nonexist_thresh=0.5, alpha_at_birth=0.5) -> None:
        self._targets: Dict[int, Target] = dict()
        self.def_alpha = default_alpha
        self.alpha_at_birth = alpha_at_birth
        self.alpha_nonexist_thresh = alpha_nonexist_thresh
        self.state_size = 3
        self._next_available_id = 0
        self._classes = {"blue", "yellow", "orange", "big_orange"}

    # dist matrix func in init
    def execute(self, localization: Localization, detections: List[Detection], dtime=1) -> TrackedCones:
        self._predict(localization, dtime=dtime)
        associated, unassociated_detections = self._associate(detections, distance_matrix_func=l2_norm)
        self._update(associated)  # associated - dict[target_id: Detection]
        self._spawn_tracks(unassociated_detections)  # unassociated detection - List[Detection]
        self._manage_tracks()  # Only deletes tracks. Does not change confidence (for now).
        return self.get_tracks()


    def _manage_tracks(self):
        track_ids = list(self._targets.keys())
        for curr_id in track_ids:
            track = self._targets[curr_id]
            updated_track = Target(id=track.id, cone_class=track.cone_class, state=track.state,
                                   existence_probability=track.existence_probability)
            self._targets[curr_id] = updated_track

            if self._targets[curr_id].existence_probability < self.alpha_nonexist_thresh:
                del self._targets[curr_id]

    def get_tracks(self):
        return TrackedCones(cone_dict=self._targets)

    def _spawn_tracks(self, detections: List[Detection]):
        num_new_ids = len(detections)
        available_ids = range(self._next_available_id, self._next_available_id + num_new_ids).__iter__() # TODO: can make longint
        self._next_available_id = self._next_available_id + num_new_ids

        for det in detections:
            state = get_detection_state(det)
            new_id = available_ids.__next__()
            probability_to_exist = self.alpha_at_birth
            target = Target(id=new_id, state=state, existence_probability=probability_to_exist, cone_class=det.cone_class)
            self._targets[new_id] = target


    def _associate(self, detections: List[Detection], distance_matrix_func=l2_norm) -> (Dict[int, Detection], List[Detection]):
            predicted = self._targets

            if len(detections) == 0 or len(predicted) == 0:
                return self._no_detection_or_prediction_asociation(detections)

            # l2 norm between every possible detection and prediction pair
            # The shape will be (large matrix rows, short matrix rows, number of values)
            dist_msg = distance_matrix_func(detections, predicted, filter_color=False)
            predictions_ind, detections_ind = linear_sum_assignment(dist_msg.distance_matrix)  # dist_msg[i, j] - distance between prediction i and detection j

            # Find the corresponding rows in the original arrays
            det_mat, pred_mat, id_to_detection = dist_msg.detection_mat, dist_msg.pred_matrix, dist_msg.orgDetIdx_to_detection
            assigned_tracks = self._get_assigned(detections_ind, predictions_ind, det_mat, pred_mat, predicted, id_to_detection)
            unassigned_detections = self._get_unassigned_detections(detections_ind,
                                                                    det_mat, id_to_detection)
            return assigned_tracks, unassigned_detections

    def _get_unassigned_detections(self, detection_index: list, det_mat: np.ndarray, id_to_detection: Dict[int, Detection]) -> List[Detection]:
        # all the rows/columns that are unassigned
        unassigned_mat_idx = list(set(np.arange(det_mat.shape[0])) - set(detection_index))
        det_unmatched = det_mat[unassigned_mat_idx]
        detections = []
        for det_idx in unassigned_mat_idx:
            detections.append(self._numpy_detection_to_Detection(det_mat[det_idx], id_to_detection[det_idx].cov_mat))

        return detections



    def _numpy_detection_to_Detection(self, detection: np.ndarray, det_cov_mat: np.ndarray) -> Detection:
        """
        Returns a Detection object from a row in a numpy matrix.

        Args:
            detection: a row of a detection matrix organized s.t. detection[0] = cone_class, and detection[1:3] are xyz coordinates
            detections: The original list of detections. needed to get the covariances

        Returns:
            a Detection object
        """
        x, y, z = detection[1:].astype(np.float32)
        # x, y, z = detection[1], detection[2], detection[3]
        cone_class = detection[0]
        # cone_class = "blue"  # TODO: add color
        if cone_class not in self._classes:
            raise Exception(f"Tracker Error: Invalid Cone Class: {cone_class} \nValid classes are: {self._classes}")

        assert type(x) == type(y) == type(z) and type(x) in {np.float16, np.float32, np.float64, float}

        return Detection(cam_x=x, cam_y=y, cam_z=z, cone_class=cone_class, cov_mat=det_cov_mat)

    def _get_unassigned_predictions(self, assigned_tracks: Dict[int, Target], predicted_obj: Dict[int, Target]) -> Dict[
        int, Target]:
        unassigned_ids = set(predicted_obj.keys()) - set(assigned_tracks.keys())
        unassigned = dict()
        for id in unassigned_ids:
            unassigned[id] = predicted_obj[id]
        return unassigned

    def _no_detection_or_prediction_asociation(self, detections: List) -> (Dict, List):
        if len(detections) == 0:
            assigned_detections, unassigned_detections = {}, []
            return assigned_detections, unassigned_detections
        else:
            unassigned_detections = detections
            assigned_detections = {}
            return assigned_detections, unassigned_detections

    def _get_assigned(self, det_ind: list, pred_ind: list, det_mat: np.ndarray, pred_mat: np.ndarray,
                      predicted_obj: Dict[int, Target], detections: Dict[int, Detection]) -> Dict[int, Detection]:
        pred_matched = pred_mat[pred_ind]
        det_matched = det_mat[det_ind]
        ordered_ids = pred_matched[:, 0].reshape(-1, 1)
        new_ass_mat = np.concatenate([ordered_ids, det_matched], axis=1)
        ordered_ids_ind = ordered_ids.reshape(-1)
        pred_ind2det_ind = dict(zip(tuple(ordered_ids_ind.tolist()), det_ind))
        return self._assigned_detection_matrix_to_dict(new_ass_mat, detections, pred_ind2det_ind)

    def _id_to_detection_covariance_map(self, assignment_matrix: np.ndarray, detections: Dict[int, Detection]) -> Dict[int, np.ndarray]:
        id_to_cov = {}
        detection_set = set(detections)
        for row in assignment_matrix:

            # Search for a row with the same state to find the covariance matrix # TODO: make a general case for covariances. This is very wrong
            detected = None
            for det in detection_set:
                if det.cam_x == row[1] and det.cam_y == row[2] and det.cam_z == row[3]:
                    id_to_cov[row[0]] = det.cov_mat
                    detected = det
                    break

            # remove found detections so they are not searched twice
            if detected is not None:
                detection_set.remove(detected)
        return id_to_cov


    def _assigned_detection_matrix_to_dict(self, assigned_mat: np.ndarray, id2org_detection: Dict[int, Detection], pred_ind2det_ind: Dict[int, int]) -> Dict[int, Detection]:
        detection_dict = {}
        for detection_row in assigned_mat:
            pred_id = detection_row[0]
            org_det_ind = pred_ind2det_ind[pred_id]
            detection_obj = self._numpy_detection_to_Detection(detection_row[1:], id2org_detection[org_det_ind].cov_mat)
            detection_dict[pred_id] = detection_obj
        return detection_dict

    def _assigned_matrix_to_Target(self, assignment_matrix, predicted_obj) -> Dict[int, Target]:
        id2new_target = dict()
        for row in assignment_matrix:
            new_id = row[0]
            state_x = assignment_matrix[-self.state_size:]
            state_cov = predicted_obj[new_id].state.P
            new_state = State(x=state_x, P=state_cov)
            new_target = Target(id=new_id, cone_class=predicted_obj[new_id].cone_class, state=new_state,
                                existence_probability=self.def_alpha)
            id2new_target[new_id] = new_target
        return id2new_target

    # TODO: Think about organizing dtime
    def _predict(self, localization: Localization, dtime=1) -> Dict[int, Target]:
        """
        Calculates prediction step

        Returns: new Dict[int: Target] of predictions. Does not modify input.

        """
        F, u, Q = self._prediction_params(localization, dtime)

        predictions = dict()
        for target in self._targets.values():
            new_loc = F @ target.state.x[:2] + u
            new_cov = F @ target.state.P @ F.T + Q
            new_state = State(x=new_loc, P=new_cov)
            new_existence_prob = self._calc_alpha(target)
            predictions[target.id] = Target(id=target.id, existence_probability=new_existence_prob, state=new_state,
                                            cone_class=target.cone_class)

        self._targets = predictions

    def _prediction_params(self, localization: Localization, dtime):
        F = np.array([[np.cos(localization.theta_ego), -np.sin(localization.theta_ego)],
                      [np.sin(localization.theta_ego), np.cos(localization.theta_ego)]])
        u = np.array([localization.x_ego, localization.y_ego])
        Q = dtime * np.array([[localization.qxx ** 2, 0], [0, localization.qyy ** 2]])

        return F, u, Q

    def _initialize_tracks(self, detections: List[Detection]):
        """
        Populates self._targets with an initial list of tracks. DO NOT USE
        Args:
            tracks: A List of detections

        Returns: None. Populates internal variable
        """
        if len(self._targets) != 0:
            raise Exception("Tracks already initialized")

        for idx, detection in enumerate(detections):
            det_state = get_detection_state(detection)
            tar_id, tar_state = idx, det_state
            tar_existence_probability = self.def_alpha
            target = Target(cone_class=detection.cone_class, id=tar_id, state=tar_state,
                            existence_probability=tar_existence_probability)
            self._targets[target.id] = target

    def _calc_alpha(self, target: Target) -> float:
        """
        Calculates survival probability of track. ATM uses a constant alpha (self.def_alpha)
        Args:
            target: A track for which to calculate survival probability

        Returns: The survival probability for the track
        """
        return self.def_alpha * target.existence_probability

    def _update(self, associated_detections: Dict[int, Detection], R_noise_multiplier=0):
        for key, detection in associated_detections.items():  # key -> target_id
            key = int(key)
            target = self._targets[key]
            detection_state = get_detection_state(detection)

            # get Kalman gain and noise matrix
            R_noise = detection.cov_mat  # np.array([[random.random()*R_noise_multiplier + 1, 0], [0, random.random()*R_noise_multiplier + 1]])
            K = target.state.P @ np.linalg.inv(target.state.P + R_noise)

            # covariance estimate
            new_cov = (np.eye(2) - K) @ target.state.P

            # state estimate (x represents the whole state. Not just x coordinate)
            y_hat = self._3d_state_to_2d(detection_state.x) - self._3d_state_to_2d(target.state.x)
            new_x_est = self._3d_state_to_2d(target.state.x) + K @ y_hat

            # create new target
            new_state = State(x=new_x_est, P=new_cov)
            new_target = Target(id=target.id, cone_class=target.cone_class, existence_probability=self.def_alpha, state=new_state)

            # Update target
            self._targets[key] = new_target

    def _3d_state_to_2d(self, state_3d):
        assert state_3d.shape in {(3,), (2,)}
        return state_3d[:2]

    def _update_inner_class(self, updated_dict: Dict[int, Target]) -> None:
        for key in updated_dict.keys():
            self._targets[key] = updated_dict[key]

    def _get_norm_function(self, norm_str: str):
        if norm_str in self._distance_functions.keys():
            return self._distance_functions[norm_str]
        raise Exception(f"INVALID VALUE for 'dist_mat_func_to_use'. Legal values are {list(self._distance_functions.keys())}")


    # def _get_update_params(self, localization: Localization):
    #     r_polar = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #     j_cart

    # cone predictions
    # predictions = {i: {'cov': None, 'pred': None} for i in detections[:, 0]}
    # for row in detections:  # for every cone
    #     predictions[row[0]]['pred'] = F @ row[1:] + u
    #     predictions[row[0]]['cov'] = F @ P @ F.T + Q
