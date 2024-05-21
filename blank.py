import numpy as np

# Function to calculate the average validation error for each model
def calculate_average_validation_error(validation_errors):
    avg_errors = {model: np.mean(errors) for model, errors in validation_errors.items()}
    return avg_errors

# Function to find the best model with the lowest average validation error
def find_best_model(avg_errors):
    best_model = min(avg_errors, key=avg_errors.get)
    return best_model

# Main function to perform the calculations
def main():
    # === Insert your validation and test errors here ===
    # Validation errors for inner folds for outer fold 1
    validation_errors_outer1 = {
        'Model 1': [0.12, 0.21, 0.22, 0.23, 0.15],
        'Model 2': [0.30, 0.11, 0.15, 0.30, 0.28],
        'Model 3': [0.21, 0.14, 0.26, 0.17, 0.26]
    }
    test_error_outer1 = {
        'Model 1': 0.24,
        'Model 2': 0.17,
        'Model 3': 0.22
    }

    # Validation errors for inner folds for outer fold 2
    validation_errors_outer2 = {
        'Model 1': [0.28, 0.18, 0.19, 0.27, 0.12],
        'Model 2': [0.16, 0.20, 0.27, 0.30, 0.25],
        'Model 3': [0.13, 0.16, 0.21, 0.17, 0.13]
    }
    test_error_outer2 = {
        'Model 1': 0.19,
        'Model 2': 0.16,
        'Model 3': 0.25
    }
    # === End of dataset insertion ===

    # Calculate the average validation errors for each model
    avg_errors_outer1 = calculate_average_validation_error(validation_errors_outer1)
    avg_errors_outer2 = calculate_average_validation_error(validation_errors_outer2)

    # Find the best model for each outer fold
    best_model_outer1 = find_best_model(avg_errors_outer1)
    best_model_outer2 = find_best_model(avg_errors_outer2)

    # Get the test errors of the best models
    test_error_best_model_outer1 = test_error_outer1[best_model_outer1]
    test_error_best_model_outer2 = test_error_outer2[best_model_outer2]

    # Calculate the generalization error
    test_errors = [test_error_best_model_outer1, test_error_best_model_outer2]
    generalization_error = np.mean(test_errors)

    print(f"Best Model for Outer Fold 1: {best_model_outer1}, Test Error: {test_error_best_model_outer1:.3f}")
    print(f"Best Model for Outer Fold 2: {best_model_outer2}, Test Error: {test_error_best_model_outer2:.3f}")
    print(f"Generalization Error: {generalization_error:.3f}")

if __name__ == "__main__":
    main()
