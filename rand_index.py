#Question 7 spring 2023

from itertools import combinations

def create_cluster_map(clusters):
    """
    Create a dictionary to map observations to clusters.

    Parameters:
    - clusters (list of sets): List where each set contains the indices of observations in that cluster.

    Returns:
    - dict: A dictionary mapping each observation to its cluster ID.
    """
    cluster_map = {}
    for cluster_id, cluster in enumerate(clusters):
        for item in cluster:
            cluster_map[item] = cluster_id
    return cluster_map

def calculate_rand_index(cluster_map_1, cluster_map_2, num_observations):
    """
    Calculate the Rand Index between two clusterings.

    Parameters:
    - cluster_map_1 (dict): A dictionary mapping each observation to its cluster ID in the first clustering.
    - cluster_map_2 (dict): A dictionary mapping each observation to its cluster ID in the second clustering.
    - num_observations (int): The total number of observations.

    Returns:
    - float: The Rand Index between the two clusterings.
    """
    TP = TN = FP = FN = 0
    all_pairs = list(combinations(range(1, num_observations + 1), 2))

    for i, j in all_pairs:
        in_same_cluster_1 = cluster_map_1[i] == cluster_map_1[j]
        in_same_cluster_2 = cluster_map_2[i] == cluster_map_2[j]

        if in_same_cluster_1 and in_same_cluster_2:
            TP += 1
        elif not in_same_cluster_1 and not in_same_cluster_2:
            TN += 1
        elif in_same_cluster_1 and not in_same_cluster_2:
            FP += 1
        elif not in_same_cluster_1 and in_same_cluster_2:
            FN += 1

    rand_index = (TP + TN) / (TP + TN + FP + FN)
    return rand_index

def main():
    # Example data (replace these values with your actual data)
    # Define the clusters from the first clustering (Q)
    Q_clusters = [
        {3, 4, 7},      # Cluster 1
        {6, 8, 9, 1},   # Cluster 2
        {5, 10, 2}      # Cluster 3
    ]

    # Define the ground-truth clusters (Z)
    Z_clusters = [
        {1, 2, 3, 4, 5, 6},   # Cluster 1
        {7, 8, 9, 10}         # Cluster 2
    ]

    # Number of observations (replace with the actual number of observations)
    num_observations = 10

    # Create cluster maps
    Q_map = create_cluster_map(Q_clusters)
    Z_map = create_cluster_map(Z_clusters)

    # Calculate the Rand Index
    rand_index = calculate_rand_index(Q_map, Z_map, num_observations)

    # Print the result
    print(f"Rand Index: {rand_index:.3f}")

if __name__ == "__main__":
    main()
