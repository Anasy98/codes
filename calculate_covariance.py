#Question 2 Spring 2023

import numpy as np

def calculate_covariance(correlation, std_dev_x, std_dev_y):
    """
    Calculate the covariance between two variables given their correlation and standard deviations.

    Parameters:
    - correlation (float): The correlation between the two variables.
    - std_dev_x (float): The standard deviation of the first variable.
    - std_dev_y (float): The standard deviation of the second variable.

    Returns:
    - float: The covariance between the two variables.
    """
    covariance = correlation * std_dev_x * std_dev_y
    return covariance

def main():
    # Example data (replace these values with your actual data)
    # Correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.48, -0.14, 0.15, -0.06],
        [0.48, 1.0, -0.14, 0.19, 0.19],
        [-0.14, -0.14, 1.0, 0.12, 0.0],
        [0.15, 0.19, 0.12, 1.0, 0.26],
        [-0.06, 0.19, 0.0, 0.26, 1.0]
    ])
    
    # Standard deviations
    std_devs = [1.55, 2.1, 3.0, 3.78, 1.8]  # Replace with actual standard deviations of your attributes
    
    # Indices of the attributes (replace these with the indices of the attributes you are interested in)
    index_x = 0  # Index for x1
    index_y = 3  # Index for x4
    
    # Extract the relevant correlation and standard deviations
    correlation = correlation_matrix[index_x, index_y]
    std_dev_x = std_devs[index_x]
    std_dev_y = std_devs[index_y]
    
    # Calculate the covariance
    covariance = calculate_covariance(correlation, std_dev_x, std_dev_y)
    
    # Print the result
    print(f"The covariance between attribute x{index_x+1} and attribute x{index_y+1} is approximately {covariance:.2f}")

if __name__ == "__main__":
    main()
