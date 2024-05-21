import numpy as np

# Given data
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])

# Step 1: Standardize the input data
mean_x = np.mean(x)
std_x = np.std(x)
x_standardized = (x - mean_x) / std_x

# Regularization parameter
lambda_reg = 0.7

# Step 2: Add a column of ones for the intercept term
X = np.vstack([np.ones(len(x_standardized)), x_standardized]).T

# Closed-form solution for ridge regression
identity_matrix = np.eye(X.shape[1])
identity_matrix[0, 0] = 0  # We don't regularize the intercept term
w = np.linalg.inv(X.T @ X + lambda_reg * identity_matrix) @ X.T @ y

# Extract the coefficients
w0, w1 = w

print("Intercept (w0):", w0)
print("Slope (w1):", w1)

# Step 3: Make a prediction for x = -1.3
x_new = -1.3
x_new_standardized = (x_new - mean_x) / std_x
y_pred = w0 + w1 * x_new_standardized

print(f"Prediction for x = {x_new}:", y_pred)
