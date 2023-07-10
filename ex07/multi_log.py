#!/usr/bin/env python3
import sys
sys.path.append('../')  # noqa: E402
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ex06.my_logistic_regression import MyLogisticRegression as MyLR
from mono_log import data_splitter


MAX_ITER = 10 ** 6
ALPHA = 0.0001

# Load the dataset
#  Features
x = pd.read_csv("solar_system_census.csv", index_col=0)
#  Labels
y = pd.read_csv("solar_system_census_planets.csv", index_col=0)
x = x.to_numpy()
y = y.to_numpy()
categories = ["The flying cities of Venus", "United Nations of Earth",
              "Mars Republic", "The Asteroids' Belt colonies"]


# Split the dataset into a training and a test set
(x_train, x_test, y_train, y_test) = data_splitter(x, y, 0.8)


zipcode = 0
# Train a logistic regression model to predict whether a citizen comes from
# this zipcode or not
print(f"Training a logistic regression classifier that can discriminate\n\
between citizens who come from {categories[zipcode]} (zipcode {zipcode})\n\
and everybody else.\n")
new_y_train = (y_train == zipcode).astype(float)
new_y_test = (y_test == zipcode).astype(float)
mylr_0 = MyLR([23.69, -0.1, -0.06, -0.66], max_iter=MAX_ITER, alpha=ALPHA)
mylr_0.fit_(x_train, new_y_train)
train_predictions_0 = mylr_0.predict_(x_train)
print(f"Training set loss: {mylr_0.loss_(new_y_train, train_predictions_0)}\n")
test_predictions_0 = mylr_0.predict_(x_test)
print(f"Test set loss: {mylr_0.loss_(new_y_test, test_predictions_0)}\n")


zipcode = 1
# Train a logistic regression model to predict whether a citizen comes from
# this zipcode or not
print(f"Training a logistic regression classifier that can discriminate\n\
between citizens who come from {categories[zipcode]} (zipcode {zipcode})\n\
and everybody else.\n")
new_y_train = (y_train == zipcode).astype(float)
new_y_test = (y_test == zipcode).astype(float)
mylr_1 = MyLR([-6.3, -0.02, 0.02, 8.87], max_iter=MAX_ITER, alpha=ALPHA)
mylr_1.fit_(x_train, new_y_train)
train_predictions_1 = mylr_1.predict_(x_train)
print(f"Training set loss: {mylr_1.loss_(new_y_train, train_predictions_1)}\n")
test_predictions_1 = mylr_1.predict_(x_test)
print(f"Test set loss: {mylr_1.loss_(new_y_test, test_predictions_1)}\n")


zipcode = 2
# Train a logistic regression model to predict whether a citizen comes from
# this zipcode or not
print(f"Training a logistic regression classifier that can discriminate\n\
between citizens who come from {categories[zipcode]} (zipcode {zipcode})\n\
and everybody else.\n")
new_y_train = (y_train == zipcode).astype(float)
new_y_test = (y_test == zipcode).astype(float)
mylr_2 = MyLR([-22.49, 0.04, 0.14, -0.26], max_iter=MAX_ITER, alpha=ALPHA)
mylr_2.fit_(x_train, new_y_train)
train_predictions_2 = mylr_2.predict_(x_train)
print(f"Training set loss: {mylr_2.loss_(new_y_train, train_predictions_2)}\n")
test_predictions_2 = mylr_2.predict_(x_test)
print(f"Test set loss: {mylr_2.loss_(new_y_test, test_predictions_2)}\n")


zipcode = 3
# Train a logistic regression model to predict whether a citizen comes from
# this zipcode or not
print(f"Training a logistic regression classifier that can discriminate\n\
between citizens who come from {categories[zipcode]} (zipcode {zipcode})\n\
and everybody else.\n")
new_y_train = (y_train == zipcode).astype(float)
new_y_test = (y_test == zipcode).astype(float)
mylr_3 = MyLR([0.11, 0.14, -0.17, -17.23], max_iter=MAX_ITER, alpha=ALPHA)
mylr_3.fit_(x_train, new_y_train)
train_predictions_3 = mylr_3.predict_(x_train)
print(f"Training set loss: {mylr_3.loss_(new_y_train, train_predictions_3)}\n")
test_predictions_3 = mylr_3.predict_(x_test)
print(f"Test set loss: {mylr_3.loss_(new_y_test, test_predictions_3)}\n")


# Calculate predictions on the entire dataset (training + test)
train_predictions = np.hstack([train_predictions_0, train_predictions_1,
                               train_predictions_2, train_predictions_3])
train_predictions = np.argmax(train_predictions, axis=1).reshape(-1, 1)

test_predictions = np.hstack([test_predictions_0, test_predictions_1,
                              test_predictions_2, test_predictions_3])
test_predictions = np.argmax(test_predictions, axis=1).reshape(-1, 1)


# Calculate and display the fraction of correct predictions over the total
# number of predictions based on the test set

correct_train = train_predictions == y_train
train_accuracy = correct_train.mean()
print(f"Training set accuracy = {train_accuracy}\n")

correct_test = test_predictions == y_test
test_accuracy = correct_test.mean()
print(f"Test set accuracy = {test_accuracy}\n")


# Plot 3 scatter plots (one for each pair of citizen features) with the
# dataset and the final prediction of the model
all_x = np.vstack([x_train, x_test])
all_y = np.vstack([y_train, y_test])
all_predictions = np.vstack([train_predictions, test_predictions])
all_correct = np.vstack([correct_train, correct_test])

sns.set(style='darkgrid')
sns.scatterplot(x=all_x[:, 0], y=all_x[:, 1], hue=all_y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

sns.scatterplot(x=all_x[:, 0], y=all_x[:, 2], hue=all_y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Height')
plt.ylabel('Bone Density')
plt.show()

sns.scatterplot(x=all_x[:, 1], y=all_x[:, 2], hue=all_y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Weight')
plt.ylabel('Bone Density')
plt.show()

# Plot a 3-D scatter plot of the dataset and our predictions
fig = plt.figure()
ax = plt.axes(projection='3d')
correct_predictions = (all_correct == 1).ravel()
colors = all_y[correct_predictions, :].astype(int)
scatter = ax.scatter(all_x[correct_predictions, 0],
                     all_x[correct_predictions, 1],
                     all_x[correct_predictions, 2],
                     c=colors, cmap="Set1", label='Correct',
                     marker='o')
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left",
                    title="Planets' Zipcodes")
ax.add_artist(legend1)

incorrect_predictions = (all_correct == 0).ravel()
colors = all_y[incorrect_predictions, :].astype(int)
ax.scatter(all_x[incorrect_predictions, 0],
           all_x[incorrect_predictions, 1],
           all_x[incorrect_predictions, 2],
           c=colors, cmap="Set1", label='Incorrect', marker='X')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Bone Density')
ax.legend()
plt.show()
