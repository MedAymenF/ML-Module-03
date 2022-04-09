import sys
sys.path.append('../')  # noqa: E402
from ex00.sigmoid import sigmoid_
import numpy as np


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
Args:
    x: has to be an numpy.array, a vector of shape m * n.
    theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
Return:
    y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
    with expected and compatible shapes.
    None: otherwise.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return sigmoid_(X @ theta)
