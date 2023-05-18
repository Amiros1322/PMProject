import numpy as np

def propagate_state(f, dt: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Runge-Kutta 4 order numerical integration.
    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    f is the derivative (or motion model)
    x is the state
    u is the input
    page 3 in Vehicle State Estimator - Yonattan Menaker
    """
    k1 = f(x, u)                                            # k_1 = x_dot(x_k, u_k)
    k2 = f(x + (dt / 2) * k1, u)                            # k_2 = x_dot(x_k + (T/2) * k_1, u_(k + 0.5))
    k3 = f(x + (dt / 2) * k2, u)                            # k_3 = x_dot(x_k + (T/2) * k_2, u_(k + 0.5))
    k4 = f(x + dt * k3, u)                                  # k_4 = x_dot(x_k + T * k_3, u_(k + 1))
    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)     # x_(k + 1) = x_k + (T/6)*(k_1 + k_2 + k_3 + k_4)
    return x_next
