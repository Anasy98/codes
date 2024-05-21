#Question 27 fall 2023

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def plot_gmm(data, means, covariances, weights, title):
    """
    Plot the GMM with the given means, covariances, and weights.
    
    Parameters:
    - data: The dataset used for generating the GMM.
    - means: A list of means for each Gaussian component.
    - covariances: A list of covariance matrices for each Gaussian component.
    - weights: A list of weights for each Gaussian component.
    - title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], s=2, label='Data points')
    
    # Plot each Gaussian component
    for mean, cov, weight in zip(means, covariances, weights):
        plot_gaussian(mean, cov)
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gaussian(mean, cov, ax=None):
    """
    Plot a single Gaussian component.
    
    Parameters:
    - mean: Mean of the Gaussian.
    - cov: Covariance matrix of the Gaussian.
    - ax: Axis to plot on. If None, creates a new axis.
    """
    if ax is None:
        ax = plt.gca()
    
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Width and height of the ellipse (2 standard deviations)
    width, height = 2 * np.sqrt(vals)
    
    # Create the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2)
    
    # Add the ellipse to the plot
    ax.add_patch(ellipse)

def generate_data():
    """
    Generate the synthetic data used for the example plot.
    """
    np.random.seed(0)
    
    # Means and covariances for the synthetic clusters
    means = [
        [-7.4, 0.3],
        [1.7, -8.4],
        [10.7, -11.5]
    ]
    covariances = [
        [[1.3, -1.5], [-1.5, 2.3]],
        [[3.0, 0.8], [0.8, 2.1]],
        [[0.6, 0.3], [0.3, 1.1]]
    ]
    weights = [0.125, 0.375, 0.5]
    
    # Generate samples
    data = []
    for mean, cov, weight in zip(means, covariances, weights):
        n_samples = int(weight * 1000)
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        data.append(samples)
    data = np.vstack(data)
    
    return data

def main():
    # Generate synthetic data (this would be replaced with your actual data)
    data = generate_data()
    
    # Define the GMM parameters (means, covariances, and weights) for each option
    option = 'C'  # Replace with 'A', 'B', 'C', or 'D' based on the exam question
    
    if option == 'A':
        means = [
            [1.7, -8.4],
            [10.7, -11.5],
            [-7.4, 0.3]
        ]
        covariances = [
            [[3.0, 0.8], [0.8, 2.1]],
            [[0.6, 0.3], [0.3, 1.1]],
            [[1.3, -1.5], [-1.5, 2.3]]
        ]
        weights = [3/8, 1/2, 1/8]
    elif option == 'B':
        means = [
            [1.7, -8.4],
            [10.7, -11.5],
            [-7.4, 0.3]
        ]
        covariances = [
            [[1.3, -1.5], [-1.5, 2.3]],
            [[0.6, 0.3], [0.3, 1.1]],
            [[3.0, 0.8], [0.8, 2.1]]
        ]
        weights = [1/8, 3/8, 1/2]
    elif option == 'C':
        means = [
            [1.7, -8.4],
            [10.7, -11.5],
            [-7.4, 0.3]
        ]
        covariances = [
            [[3.0, 0.8], [0.8, 2.1]],
            [[0.6, 0.3], [0.3, 1.1]],
            [[1.3, -1.5], [-1.5, 2.3]]
        ]
        weights = [1/8, 1/2, 3/8]
    elif option == 'D':
        means = [
            [1.7, -8.4],
            [10.7, -11.5],
            [-7.4, 0.3]
        ]
        covariances = [
            [[1.3, -1.5], [-1.5, 2.3]],
            [[3.0, 0.8], [0.8, 2.1]],
            [[0.6, 0.3], [0.3, 1.1]]
        ]
        weights = [3/8, 1/2, 1/8]
    else:
        raise ValueError("Invalid option. Choose 'A', 'B', 'C', or 'D'.")

    # Plot the GMM for the chosen option
    plot_gmm(data, means, covariances, weights, f"GMM Model - Option {option}")

if __name__ == "__main__":
    main()
