import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import jaccard_score
from sklearn.metrics import adjusted_rand_score

def cosine_similarity(matrix):
    """
    Calculate the cosine similarity between rows of the binary matrix.
    
    Parameters:
    matrix (np.array): A binary matrix with observations as rows and features as columns.
    
    Returns:
    np.array: Cosine similarity matrix.
    """
    return 1 - pairwise_distances(matrix, metric='cosine')

def simple_matching_coefficient(matrix):
    """
    Calculate the simple matching coefficient between rows of the binary matrix.
    
    Parameters:
    matrix (np.array): A binary matrix with observations as rows and features as columns.
    
    Returns:
    np.array: Simple matching coefficient matrix.
    """
    n = matrix.shape[0]
    smc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            smc_matrix[i, j] = np.sum(matrix[i] == matrix[j]) / matrix.shape[1]
    return smc_matrix

def max_min_smc(smc_matrix):
    """
    Calculate the maximum and minimum Simple Matching Coefficient (SMC).
    
    Parameters:
    smc_matrix (np.array): Simple matching coefficient matrix.
    
    Returns:
    tuple: Maximum and minimum SMC.
    """
    n = smc_matrix.shape[0]
    max_smc = np.max(smc_matrix[np.triu_indices(n, k=1)])
    min_smc = np.min(smc_matrix[np.triu_indices(n, k=1)])
    return max_smc, min_smc

def jaccard_index(matrix):
    """
    Calculate the Jaccard index between rows of the binary matrix.
    
    Parameters:
    matrix (np.array): A binary matrix with observations as rows and features as columns.
    
    Returns:
    np.array: Jaccard index matrix.
    """
    n = matrix.shape[0]
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jaccard_matrix[i, j] = jaccard_score(matrix[i], matrix[j])
    return jaccard_matrix

def rand_index(true_labels, predicted_labels):
    """
    Calculate the Rand index between two clusterings.
    
    Parameters:
    true_labels (list): The ground truth labels.
    predicted_labels (list): The predicted cluster labels.
    
    Returns:
    float: Rand index.
    """
    return adjusted_rand_score(true_labels, predicted_labels)

def main():
    # === Insert your binary dataset here ===
    # Example binary dataset (replace with your actual data)
    matrix = np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])

    # Example true labels for Rand index calculation
    true_labels = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2']
    predicted_labels = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2']  # Replace with actual clustering results if available

    # Calculate the similarity measures and indices
    cosine_sim = cosine_similarity(matrix)
    smc = simple_matching_coefficient(matrix)
    max_smc, min_smc = max_min_smc(smc)
    jaccard = jaccard_index(matrix)
    rand_idx = rand_index(true_labels, predicted_labels)

    # Set print options for better formatting
    np.set_printoptions(precision=2, suppress=True, linewidth=100)

    # Print the results with improved formatting
    print("Cosine Similarity Matrix:\n", cosine_sim)
    print("\nSimple Matching Coefficient Matrix:\n", smc)
    print(f"\nMax SMC: {max_smc:.2f}")
    print(f"Min SMC: {min_smc:.2f}")
    print("\nJaccard Index Matrix:\n", jaccard)
    print("\nRand Index:", rand_idx)

if __name__ == "__main__":
    main()
