import numpy as np
from tqdm import tqdm

from playground.bguav.vehicle_state_estimator.src.unscented_transform import UTParams, UTUtils


def _weighted_cross_correlation(X: np.ndarray, x: np.ndarray, Z:np.ndarray, z:np.ndarray, weights: np.ndarray) -> np.ndarray:
    A = X - x
    B = Z - z
    n = len(weights)
    outers = [np.dot(A[:, [i]], B[:, [i]].T) for i in range(n)]
    result = sum([w * outer for w, outer in zip(weights, outers)])
    return result


def _weighted_sum(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros((X.shape[0], 1))
    for i in range(len(weights)):
        result += weights[i] * X[:, [i]]
    return result


def _get_full_rank_random_matrix(rank: int) -> np.ndarray:
    F = np.random.rand(rank,rank)
    while np.linalg.matrix_rank(F) < rank:
        F = np.random.rand(rank,rank)
    return F


def backsubs(A, B):
    """
    copy pasted from:
    https://github.com/Al-khwarizmi-780/OpenKF/blob/main/python/examples/Square_Root_Unscented_Kalman_Filter.ipynb
    """
    # x_ik = (b_ik - Sum_aij_xjk) / a_ii
    
    N = np.shape(A)[0]
    
    X = np.zeros((B.shape[0], B.shape[1]))
    
    for k in range(B.shape[1]):
        for i in range(N-1, -1, -1):
            sum_aij_xj = B[i, k]

            for j in range(N-1, i, -1):
                sum_aij_xj -= A[i, j] * X[j, k]

            X[i, k] = sum_aij_xj / A[i, i]
    
    return X


def forwardsubs(A, B):
    """
    copy pasted from:
    https://github.com/Al-khwarizmi-780/OpenKF/blob/main/python/examples/Square_Root_Unscented_Kalman_Filter.ipynb
    """
    # x_ik = (b_ik - Sum_aij_xjk) / a_ii
    
    N = np.shape(A)[0]
    X = np.zeros((B.shape[0], B.shape[1]))
    
    for k in range(B.shape[1]):
        for i in range(N):
            sum_aij_xj = B[i, k]
            
            for j in range(i):
                sum_aij_xj -= A[i, j] * X[j, k]
                
            X[i, k] = sum_aij_xj / A[i, i]
    
    return X


def test_weighted_cross_correlation():
    N_ITERATIONS = 10000
    STATE_LEN = 5
    MEASUREMENT_LEN = 3
    n_sigma_points = 1 + 2 * STATE_LEN
    for i in tqdm(range(N_ITERATIONS), desc="Testing weighted cross correlation"):
        X = np.random.rand(STATE_LEN, n_sigma_points)
        x = np.mean(X, axis=1).reshape(-1, 1)
        Z = np.random.rand(MEASUREMENT_LEN, n_sigma_points)
        z = np.mean(Z, axis=1).reshape(-1, 1)
        weights = np.random.rand(n_sigma_points)
        correct_result = _weighted_cross_correlation(X, x, Z, z, weights)
        einsum_result = UTUtils.weighted_cross_correlation(X, x, Z, z, weights)
        assert np.allclose(correct_result, einsum_result)


def test_weighted_sum():
    N_ITERATIONS = 10000
    STATE_LEN = 5
    n_sigma_points = 1 + 2 * STATE_LEN
    for i in tqdm(range(N_ITERATIONS), desc="Testing weighted sum"):
        X = np.random.rand(STATE_LEN, n_sigma_points)
        weights = np.random.rand(n_sigma_points)
        correct_result = _weighted_sum(X, weights)
        func_result = UTUtils.weighted_sum(X, weights)
        assert np.allclose(correct_result, func_result)


def test_unscented_transform():
    N_ITERATIONS = 1000
    STATE_LEN = 6
    p = UTParams(0.5, STATE_LEN)
    for i in tqdm(range(N_ITERATIONS), desc="Testing unscented transform"):
        mean = np.random.rand(STATE_LEN,1)
        cov = np.random.rand(STATE_LEN,STATE_LEN)
        cov = np.dot(cov, cov.T)
        F = _get_full_rank_random_matrix(STATE_LEN)
        exact_mean = F.dot(mean)
        exact_cov = F.dot(cov).dot(F.T)
        L = np.linalg.cholesky(cov)
        sigma_points = UTUtils.create_sigma_points(mean, L, p.gamma)
        sigma_points_after_transform = np.dot(F, sigma_points)
        approx_mean = UTUtils.weighted_sum(sigma_points_after_transform, p.mean_weight_vector)
        approx_cov = UTUtils.weighted_correlation(sigma_points_after_transform, approx_mean, p.covariance_weight_vector)
        assert np.allclose(approx_mean, exact_mean)
        assert np.allclose(exact_cov, approx_cov)


def test_cholupdate():
    N_ITERATIONS = 1000
    STATE_LEN = 6
    p = UTParams(0.5, STATE_LEN)
    for i in tqdm(range(N_ITERATIONS), desc="Testing cholupdate"):
        mean = np.random.rand(STATE_LEN,1)
        cov = np.random.rand(STATE_LEN,STATE_LEN)
        cov = np.dot(cov, cov.T)
        F = _get_full_rank_random_matrix(STATE_LEN)
        exact_mean = F.dot(mean)
        exact_cov = F.dot(cov).dot(F.T)
        L = np.linalg.cholesky(cov)
        sigma_points = UTUtils.create_sigma_points(mean, L, p.gamma)
        sigma_points_after_transform = np.dot(F, sigma_points)
        approx_mean = UTUtils.weighted_sum(sigma_points_after_transform, p.mean_weight_vector)
        approx_L = UTUtils.estimate_covariance_cholesky_decomposition(
            sigma_points_after_transform, approx_mean, p.covariance_weight_vector)
        approx_cov = np.dot(approx_L, approx_L.T)
        assert np.allclose(approx_mean, exact_mean)
        assert np.allclose(exact_cov, approx_cov)


def test_back_and_forward_sub():
    A = np.array([[1., -2., 1.],
              [0., 1., 6.],
              [0., 0., 1.]])
    b = np.array([[4.0], [-1.0], [2.0]])
    x1 = backsubs(A, b)
    x2 = UTUtils.back_slash(A, b)
    assert np.allclose(x1, x2)
    x3 = forwardsubs(A.T, b).T
    x4 = UTUtils.forward_slash(A, b.T)
    assert np.allclose(x3, x4)



if __name__ == "__main__":
    # test_weighted_sum()
    # test_weighted_cross_correlation()
    # test_unscented_transform()
    # test_cholupdate()
    test_back_and_forward_sub()
