# Question 12 fall 2023

import numpy as np

# Function to standardize the input data
def standardize_data(x):
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    z = (x - mu_x) / sigma_x
    return z, mu_x, sigma_x

# Function to compute the regularized linear regression parameters
def regularized_linear_regression(z, y, lambda_reg):
    # Create the design matrix with a column of ones for the intercept
    X = np.vstack((np.ones(len(z)), z)).T
    
    # Compute the regularization matrix
    I = np.eye(X.shape[1])
    I[0, 0] = 0  # Do not regularize the intercept term
    
    # Compute the regularized normal equation
    XTX = X.T @ X
    XTy = X.T @ y
    w_star = np.linalg.inv(XTX + lambda_reg * I) @ XTy
    
    return w_star

# Function to make a prediction for a given input
def predict(w_star, x, mu_x, sigma_x):
    z = (x - mu_x) / sigma_x  # Standardize the input
    y_hat = w_star[0] + w_star[1] * z  # Prediction using the model
    return y_hat

# Main function to handle the entire process
def main():
    # Given data (Example dataset)
    y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])
    x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
    
    # Regularization parameter
    lambda_reg = 0.7
    
    # Standardize the input data
    z, mu_x, sigma_x = standardize_data(x)
    
    # Compute the regularized linear regression parameters
    w_star = regularized_linear_regression(z, y, lambda_reg)
    
    # Make the prediction for x2
    x2 = -1.3
    y_hat = predict(w_star, x2, mu_x, sigma_x)
    
    # Print the results
    print(f"Standardized x values: {z}")
    print(f"Mean of x: {mu_x}")
    print(f"Standard deviation of x: {sigma_x}")
    print(f"Computed weights (w*): {w_star}")
    print(f"The prediction for x2 = {x2} is: {y_hat:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
