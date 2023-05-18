def add_new_target_to_dict(tar_id: int, tar_isblue: bool, tar_state_arr: np.ndarray, tar_cov_mat: np.ndarray,
                           tar_exist_prob: float, dict_to_update: Dict):
    new_state = State(x=tar_state_arr, P=tar_cov_mat)
    new_target = Target(id=tar_id, is_blue=tar_isblue, state=new_state,
                        existence_probability=tar_exist_prob)
    dict_to_update[tar_id] = new_target