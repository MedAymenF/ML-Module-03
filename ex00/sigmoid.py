import numpy as np


def sigmoid_(x):
    """Compute the sigmoid of a vector.
Args:
    x: has to be an numpy.array, a vector
Return:
    The sigmoid value as a numpy.array.
    None otherwise.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or not x.size\
            or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a vector.")
        return None
    return (1 / (1 + np.exp(-x))).reshape(-1, 1)
