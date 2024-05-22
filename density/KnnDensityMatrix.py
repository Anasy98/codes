#QUESTION 5 SPRING 2023
# CALCLATION K N DENSITY 

import numpy as np

# Function to calculate the K-nearest neighborhood density
def k_nearest_density(distance_matrix, observation_index, k):
    """
    Calculate the K-nearest neighborhood density for a given observation.

    Parameters:
    distance_matrix (numpy.ndarray): The distance matrix containing distances between observations.
    observation_index (int): The index of the observation for which to calculate the density.
    k (int): The number of nearest neighbors to consider.

    Returns:
    float: The K-nearest neighborhood density for the specified observation.
    """
    # Get distances from the observation to all other observations
    distances = distance_matrix[observation_index]
    
    # Exclude the observation itself by setting its distance to infinity
    distances[observation_index] = np.inf
    
    # Find the indices of the K nearest neighbors
    nearest_indices = np.argsort(distances)[:k]
    
    # Calculate the density using the provided formula
    density = k / np.sum(distances[nearest_indices])
    
    return density

# Example usage
def example_usage():
    # EDIT FIRST HERE. ADD YOUR TABLE. !!!!!!!!!!!!!!!
    # Given distance matrix (Table 2)
    distance_matrix = np.array([
        [0.0, 1.3, 4.1, 3.8, 4.5, 2.4, 3.2, 2.7, 3.0, 3.9],
        [1.3, 0.0, 3.2, 3.1, 4.7, 2.3, 2.6, 2.2, 2.7, 4.2],
        [4.1, 3.2, 0.0, 0.4, 4.9, 2.7, 1.1, 1.6, 2.4, 4.8],
        [3.8, 3.1, 0.4, 0.0, 4.6, 2.5, 0.9, 1.3, 2.1, 4.5],
        [4.5, 4.7, 4.9, 4.6, 0.0, 3.1, 4.4, 3.7, 2.8, 2.3],
        [2.4, 2.3, 2.7, 2.5, 3.1, 0.0, 1.8, 1.2, 0.9, 2.8],
        [3.2, 2.6, 1.1, 0.9, 4.4, 1.8, 0.0, 1.0, 1.7, 4.1],
        [2.7, 2.2, 1.6, 1.3, 3.7, 1.2, 1.0, 0.0, 1.1, 3.6],
        [3.0, 2.7, 2.4, 2.1, 2.8, 0.9, 1.7, 1.1, 0.0, 2.9],
        [3.9, 4.2, 4.8, 4.5, 2.3, 2.8, 4.1, 3.6, 2.9, 0.0]
    ])

    # EDIT FINAL . ADD YOUR OBSERVATION COLUMN AND K . !!!!!!!!!!!!!!!!!!!!!

    # Calculate the density for observation o1 (index 0) with K = 3
    observation_index = 0
    k = 3
    density_o1_k3 = k_nearest_density(distance_matrix, observation_index, k)
    print(f"Density for observation o1 with K={k}: {density_o1_k3:.3f}")

# Run the example usage function
example_usage()
