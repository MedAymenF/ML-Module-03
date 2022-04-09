import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
    """Compute the logistic loss value.
Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
Return:
    The logistic loss value as a float.
    None on any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or\
            y_hat.shape[1] != 1 or not y_hat.size or\
            not np.issubdtype(y_hat.dtype, np.number):
        print("y_hat has to be an numpy.array, a vector of shape m * 1.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat must have the same shape.')
        return None
    if not isinstance(eps, float):
        print("eps has to be a float.")
        return None
    loss = -(y.T @ np.log(y_hat + eps) +
             (1 - y).T @ np.log(1 - y_hat + eps)) / y.shape[0]
    return float(loss)
