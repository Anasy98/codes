#Question 4 Spring 2021

import numpy as np

def compute_principal_component(x, mu, V):
    """
    Compute the coordinates in the principal component space for a given observation.
    
    Parameters:
    x (np.array): The observation vector.
    mu (np.array): The mean vector.
    V (np.array): The matrix of principal component vectors.
    
    Returns:
    np.array: The coordinates in the principal component space.
    """
    x_centered = x - mu
    b = V.T @ x_centered
    return b

def main():
    # === Insert your data here ===
    # The observation vector
    x = np.array([15.5, 59.2, 1.4, 1438.0, 5.3])
    # The mean vector
    mu = np.array([12.9, 58.2, 1.7, 1436.8, 4.1])
    # The matrix of principal component vectors
    V = np.array([
        [-0.5939, 0.2906, 0.0, 0.0621, 0.6652],
        [-0.6521, 0.0759, 0.0, 0.3813, 0.0],
        [0.2028, -0.5105, 0.0, 0.4508, 0.0],
        [-0.3696, -0.5414, 0.0, -0.7244, 0.0],
        [-0.2102, -0.5967, 0.0, 0.3503, 0.3467]
    ])
    # === End of data insertion ===

    # Compute the coordinates in the principal component space
    b = compute_principal_component(x, mu, V)

    # Print the results
    print("Coordinates in the principal component space (b):")
    print(b)

if __name__ == "__main__":
    main()
