#Question 21 Fall 2020

import numpy as np
from sklearn.linear_model import LogisticRegression

# Define your own observations here.
# Replace these example values with your actual observations.
# Example: X = np.array([[x2_1], [x2_2], [x2_3], [x2_4], [x2_5], [x2_6]])
# Example: y = np.array([y1, y2, y3, y4, y5, y6])

# Feature x2 values for 6 observations (replace with your data)
X = np.array([
    [5.1],  # x2 value for observation 1
    [4.9],  # x2 value for observation 2
    [4.7],  # x2 value for observation 3
    [4.6],  # x2 value for observation 4
    [5.0],  # x2 value for observation 5
    [5.4]   # x2 value for observation 6
])

# Output x5 values for 6 observations (0 for male, 1 for female)
# Replace these values with your actual class labels
y = np.array([0, 0, 1, 1, 0, 1])

# Adding the intercept term (constant feature)
X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize and fit the logistic regression model
model = LogisticRegression(fit_intercept=False).fit(X_intercept, y)

# Get the weights
weights = model.coef_[0]
print("Weights: ", weights)
