import numpy as np
import pandas as pd


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """Compute confusion matrix to evaluate the accuracy of a classification.
Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return\
 a pandas DataFrame
    instead of a numpy array. (default=False)
Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to\
 df_option value.
    None if any error.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or not y.size:
        print("y has to be an numpy.array, a vector.")
        return None
    if not isinstance(y_hat, np.ndarray) or not y_hat.size:
        print("y_hat has to be an numpy.array, a vector.")
        return None
    if labels is not None and not isinstance(labels, list):
        print("labels has to be a list.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat have different shapes.')
        return None
    if labels is None:
        labels = sorted(list(set(np.concatenate((y, y_hat)).ravel())))
    else:
        labels = sorted(labels)
    cols = []
    for label in labels:
        value_counts = dict(zip(labels, [0] * len(labels)))
        idx = y_hat == label
        correct_labels = y[idx]
        unique, counts = np.unique(correct_labels, return_counts=True)
        correct_labels_counts = dict(zip(unique, counts))
        value_counts.update((label, correct_labels_counts[label])
                            for label in
                            value_counts.keys() & correct_labels_counts.keys())
        col = np.array(list(value_counts.values())).reshape(-1, 1)
        cols.append(col)
    confusion_matrix = np.hstack(cols)
    if df_option:
        return pd.DataFrame(confusion_matrix, columns=labels, index=labels)
    else:
        return confusion_matrix
