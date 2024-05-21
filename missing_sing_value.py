# Question 3 2023 spring MISSING PCA VALUESSSSS 

import numpy as np

def calculate_missing_singular_value(frobenius_norm, known_singular_values):
    """
    Calculate the missing singular value from the SVD given the Frobenius norm and known singular values.

    Parameters:
    - frobenius_norm (float): The Frobenius norm of the standardized data matrix.
    - known_singular_values (list of floats): The known singular values from the SVD.

    Returns:
    - float: The missing singular value.
    """
    squared_sum = frobenius_norm ** 2
    known_squared_sum = sum(sv ** 2 for sv in known_singular_values)
    missing_singular_value_squared = squared_sum - known_squared_sum
    missing_singular_value = np.sqrt(missing_singular_value_squared)
    return missing_singular_value

def main():
    # Given data (replace these values with your actual data)
    frobenius_norm = 2814.8909
    known_singular_values = [30.3832, 22.7730, 19.7263, 16.0724]  # σ1, σ3, σ4, σ5

    # Calculate the missing singular value σ2,2
    missing_singular_value = calculate_missing_singular_value(frobenius_norm, known_singular_values)
    
    # Print the result
    print(f"The missing singular value σ2,2 is approximately {missing_singular_value:.2f}")

if __name__ == "__main__":
    main()
