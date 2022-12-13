"""
Python file for running an advanced linear regression on a dataset from CSV.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# Fetch the data.
train = pd.read_csv('advanced_train.csv')
test = pd.read_csv('advanced_test.csv')

# Drop null values.
train = train.dropna()
test = test.dropna()

# Initial plot of training data.
px.scatter(x=train['x'], y=train['y'], template='seaborn').show()

# Process data for modeling.

# Set training data values.
x_train = train['x'].values
y_train = train['y'].values

# Set testing data values.
x_test = test['x'].values
y_test = test['y'].values

# Scale the data sets.
maximum = x_train.max()

x_train_norm = x_train / maximum
x_test_norm = x_test / maximum


# The model that maps x to y is:
#   f(x) = mx + c
# To train the linear regression model we wish to find the best values
# for m and c.


def initialise_params():
    """
    Initialise the parameters of the model.
    When doing a linear regression it is okay to set these as zero.
    :return:
    """
    m = 0
    c = 0
    return m, c


def prediction(x, m, c):
    """
    Predict the parameter values for the model.
    :param x:
    :param m:
    :param c:
    :return:
    """
    return np.dot(x, m) + c


def calculate_cost(x, y, m, c):
    """
    Cost function for linear regression.
    :param x:
    :param y:
    :param m:
    :param c:
    :return:
    """
    # Number of samples.
    n: int = len(x)

    # Get predicted value.
    predicted_value = prediction(x, m, c)

    # Calculate cost.
    j = np.sum(np.square(np.subtract(predicted_value, y)))
    j /= 2 * n

    return j


# Calculate Gradient Descent.
# This is an algorithm that finds the line of best fit for a training dataset.


def gradient_descent_calculation(x, y, m, c):
    """
    Method to calculate the Gradient Descent (line of best fit).
    :param x:
    :param y:
    :param m:
    :param c:
    :return:
    """
    # Number of samples.
    n = len(x)

    # Get predicted value.
    predicted_value = prediction(x, m, c)
    dc = np.sum(np.subtract(predicted_value, y)) / n
    dm = np.sum(np.multiply(np.subtract(predicted_value, y), x)) / n
    return dm, dc


# Train the linear regression.


def run_training(x, y, iterations, alpha):
    """
    Method for training the Linear Regression.
    :param x:
    :param y:
    :param iterations:
    :param alpha:
    :return:
    """
    # Initialise array for storing costs.
    costs_array = []

    # Initialise the parameters.
    m, c = initialise_params()

    for i in range(iterations):
        # Calculate the gradient and update parameters.
        dm, dc = gradient_descent_calculation(x, y, m, c)

        # Set parameters using gradients and alpha.
        m = m - (alpha * dm)
        c = c - (alpha * dc)

        # Calculate the cost.
        calculated_cost = calculate_cost(x, y, m, c)

        # Add cost to costs array.
        costs_array.append(calculated_cost)

        # Print cost at interval.
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4}: Cost {float(calculated_cost):8.2f}   ")

    return m, c, costs_array


# Run the training.
m, c, costs = run_training(x_train_norm, y_train, 20000, 0.01)
print('m, c found by gradient descent: ', m, c)

# Plot the Cost v Iteration.
fig = px.line(y=costs, title='Cost v Iteration', template='plotly_dark')
fig.update_layout(
    title_font_color='#FFFFFF',
    xaxis=dict(color='#FFFFFF', title='Iterations'),
    yaxis=dict(color='#FFFFFF', title='Cost')
)
fig.show()

# Evaluate results using the test set.
calculate_cost(x_test_norm, y_test, m, c)
