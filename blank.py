import numpy as np

def check_coordinate(x1_observed, V12, b2, mu1):
    """
    Check if the computed x1 matches the observed x1 using the given parameters.
    
    Parameters:
    x1_observed (float): The observed x1 value.
    V12 (float): The principal component loading for b2 in the first coordinate.
    b2 (float): The second coordinate in the principal component space.
    mu1 (float): The mean of the first feature.
    
    Returns:
    bool: True if the computed x1 matches the observed x1, False otherwise.
    """
    x1_computed = V12 * b2 + mu1
    return np.isclose(x1_computed, x1_observed)

def main():
    # === Insert your data here ===
    x1_observed = 15.5  # The observed x1 value
    mu1 = 12.9  # The mean of the first feature
    V12 = 1.0  # Example value, replace with actual V12 from your PCA output

    # Possible values of b from the question options
    b_values = {
        'A': -3.2,
        'B': 1.2,
        'C': 1.5,
        'D': -1.6
    }
    # === End of data insertion ===

    # Check each possible value of b
    for label, b2 in b_values.items():
        if check_coordinate(x1_observed, V12, b2, mu1):
            print(f"Option {label} is the correct answer.")
            break
    else:
        print("No matching option found.")

if __name__ == "__main__":
    main()
