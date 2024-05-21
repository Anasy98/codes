# Question 9 Spring 2023

import numpy as np

# Data points
data_points = np.array([2, 5, 8, 12, 13])

# Initial centroids
centroids = np.array([4, 10])

def kmeans_iteration(data_points, centroids):
    # Step 1: Assign each data point to the nearest centroid
    clusters = {i: [] for i in range(len(centroids))}
    for point in data_points:
        distances = np.abs(point - centroids)
        nearest_centroid = np.argmin(distances)
        clusters[nearest_centroid].append(point)
    
    # Step 2: Recalculate centroids
    new_centroids = np.array([np.mean(clusters[i]) if clusters[i] else centroids[i] for i in range(len(centroids))])
    
    # Step 3: Calculate total cost (sum of squared distances)
    total_cost = 0
    for i in range(len(centroids)):
        for point in clusters[i]:
            total_cost += (point - new_centroids[i]) ** 2
    
    return new_centroids, total_cost

# Perform one iteration of K-means
new_centroids, total_cost = kmeans_iteration(data_points, centroids)

print(f"New Centroids: {new_centroids}")
print(f"Total Cost: {total_cost}")

