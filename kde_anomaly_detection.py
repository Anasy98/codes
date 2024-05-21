import numpy as np
from scipy.stats import norm

def kernel_density_estimator(x, data, bandwidth):
    """
    Compute the KDE for a given point x using Gaussian kernels.
    
    Parameters:
    - x: The point at which to estimate the density.
    - data: The array of training observations.
    - bandwidth: The bandwidth parameter (standard deviation of the Gaussian kernel).
    
    Returns:
    - The estimated density at point x.
    """
    n = len(data)
    density = 0.15
    for xi in data:
        density += norm.pdf(x, loc=xi, scale=bandwidth)
    density /= n
    return density

def calculate_kde(x, data, bandwidth):
    """
    Calculate and print the KDE for a given point x using the provided data and bandwidth.
    
    Parameters:
    - x: The point at which to estimate the density.
    - data: The array of training observations.
    - bandwidth: The bandwidth parameter (standard deviation of the Gaussian kernel).
    """
    density = kernel_density_estimator(x, data, bandwidth)
    print(f"p(x) at x = {x}: {density:.6f}")

# Example usage
if __name__ == "__main__":
    # Example training data
    data = np.array([-0.82, 0, 5.5])
    
    # Example bandwidth (standard deviation of the Gaussian kernel)
    bandwidth = np.sqrt(0.5)
    
    # Example test point
    test_point = -0  # You can change this value as needed
    
    # Calculate and print KDE for the test point
    calculate_kde(test_point, data, bandwidth)
