import numpy as np
import scipy

class UTParams:
    def __init__(self, alpha: float, n: int, kappa: float = None):
        beta = 2
        kappa = 3 - n if kappa is None else kappa
        _lambda = alpha**2 * (n + kappa) - n      # prefix '_' added since lambda is a python keyword
        self._gamma = np.sqrt(n + _lambda)
        wm = _lambda / (n + _lambda)
        wc = wm + (1 - alpha ** 2 + beta)
        wi = 1 / (2 * (n + _lambda))
        self._weights_cov = wi * np.ones((1 + 2 * n))
        self._weights_cov[0] = wc
        self._weights_mean = wi * np.ones((1 + 2 * n))
        self._weights_mean[0] = wm

    @property
    def gamma(self):
        return self._gamma
    
    @property
    def covariance_weight_vector(self):
        return self._weights_cov

    @property
    def mean_weight_vector(self):
        return self._weights_mean


class UTUtils:
    @staticmethod
    def create_sigma_points(x: np.ndarray, L: np.ndarray, gamma: float) -> np.ndarray:
        return np.concatenate((x, x + gamma * L, x - gamma * L), axis=1)

    @staticmethod
    def weighted_sum(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.sum(weights * X, axis=1, keepdims=True)

    @staticmethod
    def estimate_covariance_cholesky_decomposition(X: np.ndarray, x: np.ndarray, weights: np.ndarray, XX: np.ndarray = None) -> np.ndarray:
        A = np.sqrt(weights[1])*(X[:, 1:] - x)
        if XX is not None:
            A = np.concatenate((A, XX), axis=1)
        # q, r ??
        q, r = np.linalg.qr(A.T)
        S_star = r.T
        C = np.sqrt(np.abs(weights[0]))*(X[:, [0]] - x)
        S = UTUtils.cholupdate(S_star, C, np.sign(weights[0]))
        return S

    @staticmethod
    def cholupdate(L: np.ndarray, W: np.ndarray, beta: int):
        """
        copy-pasted this function from:
        https://github.com/Al-khwarizmi-780/OpenKF/blob/main/python/examples/Square_Root_Unscented_Kalman_Filter.ipynb
        """
        r = np.shape(W)[1]
        m = np.shape(L)[0]
        for i in range(r):
            L_out = np.copy(L)
            b = 1.0
            for j in range(m):
                Ljj_pow2 = L[j, j]**2
                wji_pow2 = W[j, i]**2
                L_out[j, j] = np.sqrt(Ljj_pow2 + (beta / b) * wji_pow2)
                upsilon = (Ljj_pow2 * b) + (beta * wji_pow2)
                for k in range(j+1, m):
                    W[k, i] -= (W[j, i] / L[j, j]) * L[k, j]
                    L_out[k, j] = ((L_out[j, j] / L[j, j]) * L[k, j]) + (L_out[j, j] * beta * W[j, i] * W[k, i] / upsilon)
                b += beta * (wji_pow2 / Ljj_pow2)
            L = np.copy(L_out)
        return L_out

    @staticmethod
    def weighted_correlation(X: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return UTUtils.weighted_cross_correlation(X, x, X, x, weights)

    @staticmethod
    def weighted_cross_correlation(X: np.ndarray, x: np.ndarray, Z:np.ndarray, z:np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.einsum('ni,mi->nm', weights * (X - x), Z - z)

    @staticmethod
    def forward_slash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        In Matlab: b/A
        Solution of x A = b for x
        """
        return scipy.linalg.solve_triangular(A.T, b.T).T

    @staticmethod
    def back_slash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        In Matlab: A\b
        Solution of A x = b for x
        """
        return scipy.linalg.solve_triangular(A, b)


        

