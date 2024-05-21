import numpy as np
from collections import Counter

def knn_metrics(distance_matrix, labels, k):
    """
    Calculate the precision, recall, and error rate of the k-nearest neighbors (KNN) classifier.
    
    Parameters:
    distance_matrix (np.array): A square matrix containing the distances between observations.
    labels (list): A list of true labels for each observation.
    k (int): The number of nearest neighbors to consider.
    
    Returns:
    tuple: Precision, recall, and error rate of the KNN classifier.
    """
    n = distance_matrix.shape[0]
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct_predictions = 0
    
    for i in range(n):
        distances = distance_matrix[i]
        # Get the indices of the k smallest distances (excluding the distance to itself)
        nearest_neighbors = np.argsort(distances)[1:k+1]
        # Get the labels of the nearest neighbors
        nearest_labels = [labels[j] for j in nearest_neighbors]
        # Determine the most common label among the nearest neighbors
        predicted_label = Counter(nearest_labels).most_common(1)[0][0]
        # Check if the prediction is correct
        if predicted_label == labels[i]:
            correct_predictions += 1
            if predicted_label == 'C2':  # Assuming 'C2' is the positive class
                true_positive += 1
        else:
            if predicted_label == 'C2':  # False positive
                false_positive += 1
            if labels[i] == 'C2':  # False negative
                false_negative += 1
    
    # Calculate precision, recall, and error rate
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    error_rate = 1 - (correct_predictions / n)
    
    return precision, recall, error_rate

def main():
    # === Insert your distance matrix here ===
    # Example distance matrix (replace with your actual data)
    distance_matrix = np.array([
        [0.0, 2.6, 2.8, 2.1, 0.8, 1.7, 3.9, 3.8, 4.1, 4.2],
        [2.6, 0.0, 1.8, 2.3, 2.7, 1.4, 1.9, 1.7, 2.0, 2.2],
        [2.8, 1.8, 0.0, 1.2, 2.9, 2.1, 2.2, 1.6, 2.3, 2.0],
        [2.1, 2.3, 1.2, 0.0, 2.4, 1.8, 2.6, 2.5, 3.0, 2.9],
        [0.8, 2.7, 2.9, 2.4, 0.0, 2.0, 4.2, 4.0, 4.4, 4.5],
        [1.7, 1.4, 2.1, 1.8, 2.0, 0.0, 2.4, 2.6, 2.7, 3.0],
        [3.9, 1.9, 2.2, 2.6, 4.2, 2.4, 0.0, 1.2, 0.6, 1.3],
        [3.8, 1.7, 1.6, 2.5, 4.0, 2.6, 1.2, 0.0, 1.0, 0.6],
        [4.1, 2.0, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0, 0.0, 0.9],
        [4.2, 2.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.6, 0.9, 0.0]
    ])

    # === Insert your labels here ===
    # Example labels (replace with your actual labels)
    labels = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2']

    # Calculate and print the metrics for k = 1, 2, 3
    for k in range(1, 4):
        precision, recall, error_rate = knn_metrics(distance_matrix, labels, k)
        print(f"K = {k}: Precision = {precision:.2f}, Recall = {recall:.2f}, Error Rate = {error_rate:.2f}")

if __name__ == "__main__":
    main()
