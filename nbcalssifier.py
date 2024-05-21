import numpy as np

# Define the dataset
# Each row represents an observation with features and the last column is the class label
data = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # o1, class 1
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # o2, class 1
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # o3, class 1
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 2],  # o4, class 2
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2],  # o5, class 2
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 2],  # o6, class 2
    [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2],  # o7, class 2
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 2],  # o8, class 2
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 3],  # o9, class 3
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 3]   # o10, class 3
])

# Define features to use
features_indices = [0, 1, 5]  # f1, f2, f6
observation = [1, 1, 0]       # Observed feature values

# Define the classes
classes = np.unique(data[:, -1])

# Compute probabilities
def compute_naive_bayes(data, features_indices, observation, class_label):
    feature_data = data[:, features_indices]
    class_data = data[:, -1]

    # Class probability
    p_class = np.mean(class_data == class_label)
    
    # Conditional probabilities
    p_features_given_class = 1
    for feature_index, feature_value in zip(features_indices, observation):
        feature_column = data[class_data == class_label, feature_index]
        p_feature_given_class = np.mean(feature_column == feature_value)
        p_features_given_class *= p_feature_given_class

    return p_features_given_class * p_class

# Calculate the probability for each class
probabilities = {}
for cls in classes:
    probabilities[cls] = compute_naive_bayes(data, features_indices, observation, cls)

# Normalize probabilities to sum to 1
total_prob = sum(probabilities.values())
normalized_probabilities = {cls: prob / total_prob for cls, prob in probabilities.items()}

# Output the results
print("Probabilities for each class given the observation:")
for cls, prob in normalized_probabilities.items():
    print(f"Class {cls}: {prob:.3f}")

# Example: Get the probability of class 1
print(f"Probability of class 1: {normalized_probabilities[1]:.3f}")
