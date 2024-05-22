#FALL 2020 QUESTION 19
import numpy as np

# Initialize variables
y_true = np.array([2, 1, 1, 1, 2, 2, 2])
y_pred = np.array([2, 1, 2, 1, 2, 2, 2])
N = len(y_true)
weights = np.ones(N) / N

# Compute weighted error
misclassified = (y_true != y_pred)
epsilon = np.sum(weights * misclassified)

# Compute classifier weight (alpha)
alpha = 0.5 * np.log((1 - epsilon) / epsilon)

# Update weights
weights[misclassified] *= np.exp(alpha)
weights[~misclassified] *= np.exp(-alpha)

# Normalize weights
weights /= np.sum(weights)

# Print results
print(f"Weighted Error Rate (epsilon): {epsilon}")
print(f"Alpha (alpha_t): {alpha}")
print(f"Updated Weights: {weights}")
