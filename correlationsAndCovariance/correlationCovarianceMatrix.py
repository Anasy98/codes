# Spring 2023, question 2  
# put in your matrix. make sure it is square.
# 

import numpy as np
import pandas as pd

def calculate_covariance(corr_matrix, std_devs, x1_index, x2_index):
    """
    Calculate the covariance between two specified variables using their indices.

    Parameters:
    corr_matrix (numpy.ndarray): Correlation matrix.
    std_devs (list or numpy.ndarray): List or array of standard deviations.
    x1_index (int): Index of the first variable.
    x2_index (int): Index of the second variable.

    Returns:
    float: Calculated covariance between the specified variables.
    """
    # Extract the correlation value between the specified variables
    corr_value = corr_matrix[x1_index, x2_index]

    # Extract the standard deviations of the specified variables
    sigma_x1 = std_devs[x1_index]
    sigma_x2 = std_devs[x2_index]

    # Calculate the covariance using the formula: Cov[x_i, x_j] = Corr[x_i, x_j] * sigma_i * sigma_j
    covariance = corr_value * sigma_x1 * sigma_x2

    return covariance

def create_covariance_matrix(corr_matrix, std_devs):
    """
    Create the full covariance matrix from the correlation matrix and standard deviations.

    Parameters:
    corr_matrix (numpy.ndarray): Correlation matrix.
    std_devs (list or numpy.ndarray): List or array of standard deviations.

    Returns:
    pandas.DataFrame: Full covariance matrix as a DataFrame.
    """
    num_vars = len(std_devs)
    cov_matrix = np.zeros((num_vars, num_vars))

    # Populate the covariance matrix
    for i in range(num_vars):
        for j in range(num_vars):
            cov_matrix[i, j] = corr_matrix[i, j] * std_devs[i] * std_devs[j]

    # Convert to DataFrame for better visualization
    cov_matrix_df = pd.DataFrame(cov_matrix, columns=[f'x{i+1}' for i in range(num_vars)], index=[f'x{i+1}' for i in range(num_vars)])
    return cov_matrix_df

# EDIT NUMBER 1. ADD MATRIX. !!!!!!!!!!!!!

# Example usage:
# Given correlation matrix (expandable)
corr_matrix = np.array([
    [1.0, 0.48, -0.14, 0.15, -0.06, 0.5, 0.1],
    [0.48, 1.0, -0.14, 0.19, 0.19, 0.3, 0.5],
    [-0.14, -0.14, 1.0, 0.12, 0.0, 0.5, 0.2],
    [0.15, 0.19, 0.12, 1.0, 0.26, 0.1, 0.4],
    [-0.06, 0.19, 0.0, 0.26, 1.0, 0.9, 0.2],
    [0.5, 0.3, 0.5, 0.1, 0.9, 1.0, 0.6],
    [0.1, 0.5, 0.2, 0.4, 0.2, 0.6, 1.0]
])


# EDIT NUMBER 2. YOU CAN PUT STANDARD DEVIATIONS OF EACH COLUMNS (X) HERE. !!!!!!!!!!!!!


# Given standard deviations (expandable)
std_devs = np.array([1.55, 2.55, 2.49, 3.78, 0.11, 0.4, 0.2])


# EDIT NUMBER FINALLLLLLLLL. CHOOSE WHICH COLUMNS YOU WANT TO CHECK THE COVARAINCE FOR !!!!!!!!!!!!!


# Calculate covariance between x1 and x4
x1_index = 0  # Index for x1 (0-based index)
x4_index = 3  # Index for x4 (0-based index)
cov_x1_x4 = calculate_covariance(corr_matrix, std_devs, x1_index, x4_index)
print(f"The covariance between x1 and x4 is: {cov_x1_x4:.2f}")

# Generate the full covariance matrix
cov_matrix_df = create_covariance_matrix(corr_matrix, std_devs)
print("\nFull Covariance Matrix:")
print(cov_matrix_df)



# YOU NEED STANDARD DEVIATION OF COLUMNS IN QUESION. OTHER COLUMNS DONT MATTER BUT U NEED TO FILL THE STD DEVS. 

# you can use function below. stick numbers from column. one at a time. 



import numpy as np

def column_standard_deviations(data):
    # Convert the data to a NumPy array for easier manipulation
    data_array = np.array(data)
    
    # Calculate the standard deviation of each column
    std_devs = np.std(data_array, axis=0)
    
    return std_devs

# Given data
data = [
    [1.0, 0.48, -0.14, 0.15, -0.06],
    [0.48, 1.0, -0.14, 0.19, 0.19],
    [-0.14, -0.14, 1.0, 0.12, 0.0],
    [0.15, 0.19, 0.12, 1.0, 0.26],
    [-0.06, 0.19, 0.0, 0.26, 1.0],
]

# Get the standard deviations of each column
std_devs = column_standard_deviations(data)
print("Standard deviations of each column:", std_devs)
