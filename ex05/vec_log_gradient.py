import sys
sys.path.append('../')  # noqa: E402
from ex00.sigmoid import sigmoid_
import numpy as np


def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,\
 without any for-loop.
The three arrays must have compatible shapes.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
Return:
    The gradient as a numpy.array, a vector of shape n * 1,
    containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if y, x or theta is not of the expected type.
Raises:
    This function should not raise any Exception.
"""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    m = x.shape[0]
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    predictions = sigmoid_(X @ theta)
    return (X.T @ (predictions - y)) / m
