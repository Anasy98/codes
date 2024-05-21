# Question 18 2023 FALL

import numpy as np

def initialize_weights(N):
    """Initialize weights uniformly."""
    return np.ones(N) / N

def compute_weighted_error(weights, y_true, y_pred):
    """Compute the weighted error."""
    incorrect = (y_true != y_pred).astype(int)
    weighted_error = np.sum(weights * incorrect) / np.sum(weights)
    return weighted_error

def compute_alpha(weighted_error):
    """Compute the classifier weight alpha."""
    return 0.5 * np.log((1 - weighted_error) / weighted_error)

def update_weights(weights, alpha, y_true, y_pred):
    """Update the weights for the next round of boosting."""
    incorrect = (y_true != y_pred).astype(int)
    weights *= np.exp(alpha * incorrect - alpha * (1 - incorrect))
    return weights / np.sum(weights)

def main():
    # Insert your own data here
    # Example data
    N = 6
    y_true = np.array([2, 2, 1, 2, 1, 1])  # True labels
    y_pred = np.array([2, 2, 2, 1, 2, 2])  # Predicted labels after the first round

    # Initialize weights
    weights = initialize_weights(N)
    print(f"Initial weights: {weights}")

    # Compute weighted error
    weighted_error = compute_weighted_error(weights, y_true, y_pred)
    print(f"Weighted error: {weighted_error}")

    # Compute alpha
    alpha = compute_alpha(weighted_error)
    print(f"Alpha: {alpha}")

    # Update weights
    updated_weights = update_weights(weights, alpha, y_true, y_pred)
    print(f"Updated weights: {updated_weights}")

if __name__ == "__main__":
    main()
