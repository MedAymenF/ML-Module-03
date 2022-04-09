import numpy as np


def accuracy_score_(y, y_hat):
    """Compute the accuracy score.
Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
Return:
    The accuracy score as a float.
    None on any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or not y.size:
        print("y has to be an numpy.array, a vector.")
        return None
    if not isinstance(y_hat, np.ndarray) or not y_hat.size:
        print("y_hat has to be an numpy.array, a vector.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    return (y == y_hat).mean()


def precision_score_(y, y_hat, pos_label=1):
    """Compute the precision score.
Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report\
 the precision_score (default=1)
Return:
    The precision score as a float.
    None on any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or not y.size:
        print("y has to be an numpy.array, a vector.")
        return None
    if not isinstance(y_hat, np.ndarray) or not y_hat.size:
        print("y_hat has to be an numpy.array, a vector.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    tp = ((y_hat == pos_label) * (y == pos_label)).sum()
    fp = ((y_hat == pos_label) * (1 - (y == pos_label))).sum()
    return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    """Compute the recall score.
Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report\
 the precision_score (default=1)
Return:
    The recall score as a float.
    None on any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or not y.size:
        print("y has to be an numpy.array, a vector.")
        return None
    if not isinstance(y_hat, np.ndarray) or not y_hat.size:
        print("y_hat has to be an numpy.array, a vector.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    tp = ((y_hat == pos_label) * (y == pos_label)).sum()
    fn = ((y_hat != pos_label) * (y == pos_label)).sum()
    return tp / (tp + fn)


def f1_score_(y, y_hat, pos_label=1):
    """Compute the f1 score.
Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report\
 the precision_score (default=1)
Return:
    The f1 score as a float.
    None on any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or not y.size:
        print("y has to be an numpy.array, a vector.")
        return None
    if not isinstance(y_hat, np.ndarray) or not y_hat.size:
        print("y_hat has to be an numpy.array, a vector.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    precision = precision_score_(y, y_hat, pos_label)
    if precision is None:
        return None
    recall = recall_score_(y, y_hat, pos_label)
    if recall is None:
        return None
    return (2 * precision * recall) / (precision + recall)
