#Same as knn_outer_ties

import numpy as np

# Function to calculate the average error rate for each K
def calculate_average_error(errors):
    avg_errors = {k: np.mean(v) for k, v in errors.items()}
    return avg_errors

# Function to find the best K with the lowest average error rate
def find_best_k(avg_errors):
    best_k = min(avg_errors, key=avg_errors.get)
    return best_k

# Function to recalculate test error when there are ties
def recalculate_test_error(test_set_errors, best_k):
    # Count the number of misclassifications
    num_misclassifications = sum(test_set_errors[best_k])
    total_test_cases = len(test_set_errors[best_k])
    error_rate = num_misclassifications / total_test_cases
    return error_rate

# Main function to perform the calculations
def main():
    # === Insert your dataset here ===
    # Example error rates for inner folds for outer fold 1
    # Replace this with your actual data
    errors_outer1 = {
        1: [0, 0, 1, 0, 0],  # Error rates for K=1 across 5 inner folds
        3: [1, 1, 1, 1, 1],  # Error rates for K=3 across 5 inner folds
        4: [1, 1, 1, 0, 0]   # Error rates for K=4 across 5 inner folds
    }

    # Example error rates for inner folds for outer fold 2
    # Replace this with your actual data
    errors_outer2 = {
        1: [0, 0, 1, 0, 1],  # Error rates for K=1 across 5 inner folds
        3: [0, 0, 1, 0, 0],  # Error rates for K=3 across 5 inner folds
        4: [0, 0, 1, 0, 0]   # Error rates for K=4 across 5 inner folds
    }
    # === End of dataset insertion ===

    # Calculate the average error rates for each K
    avg_errors_outer1 = calculate_average_error(errors_outer1)
    avg_errors_outer2 = calculate_average_error(errors_outer2)

    # Find the best K for each outer fold
    best_k_outer1 = find_best_k(avg_errors_outer1)
    best_k_outer2 = find_best_k(avg_errors_outer2)

    # Check if there are ties in the average error rates
    if avg_errors_outer1[best_k_outer1] == avg_errors_outer2[best_k_outer2]:
        # Recalculate the test error for outer fold 2
        test_set_errors_outer2 = {
            3: [1, 1, 1, 0, 0]  # Replace with actual test set errors for outer fold 2
        }
        error_test_outer2 = recalculate_test_error(test_set_errors_outer2, 3)
    else:
        error_test_outer2 = avg_errors_outer2[best_k_outer2]

    error_test_outer1 = avg_errors_outer1[best_k_outer1]

    print(f"Best K for Outer Fold 1: {best_k_outer1}, Error Rate: {error_test_outer1:.2f}")
    print(f"Best K for Outer Fold 2: {best_k_outer2}, Error Rate: {error_test_outer2:.2f}")

if __name__ == "__main__":
    main()
