import numpy as np

class Utils:

    @staticmethod
    def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_positive_definite(a: np.ndarray) -> bool:
        return np.all(np.linalg.eigvals(a) > 0)

    @staticmethod
    def is_column_vector(a: np.ndarray) -> bool:
        return len(a.shape) == 2 and a.shape[1] == 1
