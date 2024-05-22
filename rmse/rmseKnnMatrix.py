#2020 sp4ring question 13
import pandas as pd

# Example table in a dictionary form
data = {
    "Feature(s)": [
        "none", "x1", "x2", "x3", "x4", "x5", "x1; x2", "x1; x3", "x2; x3",
        "x1; x4", "x2; x4", "x3; x4", "x1; x5", "x2; x5", "x3; x5", "x4; x5",
        "x1; x2; x3", "x1; x2; x4", "x1; x3; x4", "x2; x3; x4", "x1; x2; x5",
        "x1; x3; x5", "x2; x3; x5", "x1; x4; x5", "x2; x4; x5", "x3; x4; x5",
        "x1; x2; x3; x4", "x1; x2; x3; x5", "x1; x2; x4; x5", "x1; x3; x4; x5",
        "x2; x3; x4; x5", "x1; x2; x3; x4; x5"
    ],
    "Training RMSE": [
        1.429, 0.755, 1.421, 0.636, 0.847, 0.773, 0.640, 0.636, 0.401, 0.745,
        0.565, 0.587, 0.728, 0.449, 0.613, 0.733, 0.380, 0.541, 0.586, 0.399,
        0.448, 0.613, 0.396, 0.702, 0.407, 0.582, 0.379, 0.369, 0.400, 0.580,
        0.359, 0.315
    ],
    "Test RMSE": [
        2.02, 1.662, 1.977, 1.628, 1.636, 1.702, 1.706, 1.638, 1.912, 1.602,
        1.799, 1.890, 1.647, 1.767, 1.824, 2.155, 2.135, 1.696, 1.914, 1.954,
        1.779, 1.831, 1.828, 2.022, 2.087, 1.901, 2.168, 1.988, 2.138, 1.927,
        1.935, 2.030
    ]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Function to calculate total time for nested cross-validation
def calculate_nested_cv_time(K1, K2, S, train_time, test_time):
    """
    Calculate the total time required for a nested cross-validation procedure.
    
    Parameters:
    K1 (int): Number of folds in the outer cross-validation loop.
    K2 (int): Number of folds in the inner cross-validation loop.
    S (int): Number of different models to be evaluated.
    train_time (float): Time taken to train a single model (in seconds).
    test_time (float): Time taken to test a single model (in seconds).
    
    Returns:
    float: Total time required for the nested cross-validation procedure (in seconds).
    """
    
    # Total number of models to be trained and tested
    total_models = K1 * (K2 * S + 1)
    
    # Total training time
    total_training_time = total_models * train_time
    
    # Total testing time
    total_testing_time = total_models * test_time
    
    # Total time
    total_time = total_training_time + total_testing_time
    
    return total_time

#EDIT HEREEEEEE !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Example usage
K1 = 4  # Number of outer folds
K2 = 7  # Number of inner folds
S = 3   # Number of different model architectures
train_time = 20  # Training time for a single model (in seconds)
test_time = 1    # Testing time for a single model (in seconds)

total_time = calculate_nested_cv_time(K1, K2, S, train_time, test_time)
print(f"Total time required for nested cross-validation: {total_time / 60:.2f} minutes")
