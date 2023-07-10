#!/usr/bin/env python3
import sys
sys.path.append('../')  # noqa: E402
from ex06.my_logistic_regression import MyLogisticRegression as MyLR
import pandas as pd
import argparse
import numpy as np
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def data_splitter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)\
 into a training and a test set,
while respecting the given proportion of examples to be kept\
 in the training set.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset\
 that will be assigned to the
    training set.
Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
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
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    if not isinstance(proportion, (int, float)):
        print('proportion has to be a float.')
        return None
    if proportion < 0 or proportion > 1:
        print('proportion has to be between 0 and 1.')
        return None
    rng = default_rng(1337)
    z = np.hstack((x, y))
    rng.shuffle(z)
    x, y = z[:, :-1].reshape(x.shape), z[:, -1].reshape(y.shape)
    idx = int((x.shape[0] * proportion))
    x_train, x_test = np.split(x, [idx])
    y_train, y_test = np.split(y, [idx])
    return (x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    # Load the dataset
    #  Features
    x = pd.read_csv("solar_system_census.csv", index_col=0)
    #  Labels
    y = pd.read_csv("solar_system_census_planets.csv", index_col=0)
    x = x.to_numpy()
    y = y.to_numpy()
    categories = ["The flying cities of Venus", "United Nations of Earth",
                  "Mars Republic", "The Asteroids' Belt colonies"]

    # Parse arguments
    parser = argparse.ArgumentParser(description="A logistic regression\
 classifier that can discriminate between two classes.")
    parser.add_argument("--zipcode", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    zipcode = args.zipcode
    print(f"Training a logistic regression classifier that can discriminate\n\
between citizens who come from {categories[zipcode]} (zipcode {zipcode})\n\
and everybody else.\n")

    # Split the dataset into a training and a test set
    (x_train, x_test, y_train, y_test) = data_splitter(x, y, 0.8)

    # Generate a new numpy.array to label each citizen with 1 if it belongs to
    # the zipcode, 0 otherwise
    new_y_train = (y_train == zipcode).astype(float)
    new_y_test = (y_test == zipcode).astype(float)

    # Train a logistic regression model to predict whether a citizen comes from
    # this zipcode or not using the new_y_train labels
    mylr = MyLR(np.ones((x.shape[1] + 1, 1)), max_iter=2 * 10 ** 6)
    mylr.fit_(x_train, new_y_train)
    train_predictions = mylr.predict_(x_train)
    print(f"Training set loss: {mylr.loss_(new_y_train, train_predictions)}\n")
    test_predictions = mylr.predict_(x_test)
    print(f"Test set loss: {mylr.loss_(new_y_test, test_predictions)}\n")

    # Calculate and display the fraction of correct predictions over the total
    # number of predictions based on the test set
    train_predictions[train_predictions >= 0.5] = 1
    train_predictions[train_predictions < 0.5] = 0
    test_predictions[test_predictions >= 0.5] = 1
    test_predictions[test_predictions < 0.5] = 0

    correct_train = train_predictions == new_y_train
    train_accuracy = correct_train.mean()
    print(f"Training set accuracy = {train_accuracy}\n")

    correct_test = test_predictions == new_y_test
    test_accuracy = correct_test.mean()
    print(f"Test set accuracy = {test_accuracy}\n")

    # Plot 3 scatter plots (one for each pair of citizen features) with the
    # dataset and the final prediction of the model
    all_x = np.vstack([x_train, x_test])
    all_y = np.vstack([new_y_train, new_y_test])
    all_predictions = np.vstack([train_predictions, test_predictions])
    all_correct = np.vstack([correct_train, correct_test])

    planet = categories[zipcode]
    hue_vector = np.where(all_y, planet, 'Other zipcodes').ravel()
    style_vector = np.where(all_correct,
                            'Correctly labeled', 'Incorrectly labeled').ravel()
    sns.set(style='darkgrid')
    sns.scatterplot(x=all_x[:, 0], y=all_x[:, 1], hue=hue_vector,
                    hue_order=[planet, 'Other zipcodes'], style=style_vector,
                    markers={'Correctly labeled': 'o',
                    'Incorrectly labeled': 'X'})
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()

    sns.scatterplot(x=all_x[:, 0], y=all_x[:, 2], hue=hue_vector,
                    hue_order=[planet, 'Other zipcodes'], style=style_vector,
                    markers={'Correctly labeled': 'o',
                    'Incorrectly labeled': 'X'})
    plt.xlabel('Height')
    plt.ylabel('Bone Density')
    plt.show()

    sns.scatterplot(x=all_x[:, 1], y=all_x[:, 2], hue=hue_vector,
                    hue_order=[planet, 'Other zipcodes'], style=style_vector,
                    markers={'Correctly labeled': 'o',
                    'Incorrectly labeled': 'X'})
    plt.xlabel('Weight')
    plt.ylabel('Bone Density')
    plt.show()

    # Plot a 3-D scatter plot of the dataset and our predictions
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    correct_predictions = (all_correct == 1).ravel()
    colors = all_y[correct_predictions, :].astype(int)
    color_list = ["red", "green"]
    cmap = ListedColormap(color_list)
    scatter1 = ax.scatter(all_x[correct_predictions, 0],
                          all_x[correct_predictions, 1],
                          all_x[correct_predictions, 2], c=colors, cmap=cmap,
                          label='Correct', marker='o')
    legend1 = ax.legend(*scatter1.legend_elements(), loc="upper left",
                        title='Labels')
    ax.add_artist(legend1)

    incorrect_predictions = (all_correct == 0).ravel()
    colors = all_y[incorrect_predictions, :].astype(int)
    scatter2 = ax.scatter(all_x[incorrect_predictions, 0],
                          all_x[incorrect_predictions, 1],
                          all_x[incorrect_predictions, 2], c=colors, cmap=cmap,
                          label='Incorrect', marker='X')

    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Bone Density')
    ax.legend()
    plt.show()
