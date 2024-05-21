import numpy as np

def knn_density(distance_matrix, k):
    """
    Calculate the density for each observation in the distance matrix using k-nearest neighbors.
    
    Parameters:
    distance_matrix (np.array): A square matrix containing the distances between observations.
    k (int): The number of nearest neighbors to consider.
    
    Returns:
    np.array: Density for each observation.
    """
    n = distance_matrix.shape[0]
    densities = np.zeros(n)
    
    for i in range(n):
        distances = distance_matrix[i]
        # Get the k smallest distances (excluding the distance to itself)
        nearest_distances = np.sort(distances)[1:k+1]
        # Calculate density as the inverse of the average distance to k-nearest neighbors
        densities[i] = k / np.sum(nearest_distances)
    
    return densities

def main():
    # === Insert your distance matrix here ===
    # Example distance matrix (replace with your actual data)
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
    
    # Calculate densities for k = 1, 2, 3
    k_values = [1, 2, 3]
    for k in k_values:
        densities = knn_density(distance_matrix, k)
        print(f"\nDensities for k = {k}:")
        for i, density in enumerate(densities):
            print(f"Observation {i + 1}: {density:.4f}")

if __name__ == "__main__":
    main()
