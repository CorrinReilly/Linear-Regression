"""
Python file for creating a plot for a basic linear regression from a supplied CSV dataset.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# Get the data from the CSVs.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.dropna()

x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 1].values

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, 1].values

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Create a plot for the training set.
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, regressor.predict(x_train), color='orange')
plt.title('Linear Regression (Training Set)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Create a plot for the testing set.
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train, regressor.predict(x_train), color='yellow')
plt.title('Linear Regression (Test Set)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
