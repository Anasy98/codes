import numpy as np

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the function to calculate logistic regression probability
def logistic_regression_probability(weights, feature_value):
    # Append 1 for the intercept term
    x_input = np.array([1] + [feature_value])
    # Compute the linear combination of weights and input features
    z = np.dot(weights, x_input)
    # Compute the probability using the sigmoid function
    probability = sigmoid(z)
    return probability

# Example usage:
if __name__ == "__main__":
    # Weights in the format: [intercept, slope]
    weights_A = np.array([-0.93, 1.72])
    weights_B = np.array([-2.82, 0.0])
    weights_C = np.array([1.36, 0.4])
    weights_D = np.array([-0.65, 0.0])
    
    # Feature value for x8
    feature_x8 = 1  # Change this value based on the question

    # Calculate probabilities for each set of weights
    probability_A = logistic_regression_probability(weights_A, feature_x8)
    probability_B = logistic_regression_probability(weights_B, feature_x8)
    probability_C = logistic_regression_probability(weights_C, feature_x8)
    probability_D = logistic_regression_probability(weights_D, feature_x8)
    
    # Print the results
    print(f"Probability for Option A: {probability_A:.2f}")
    print(f"Probability for Option B: {probability_B:.2f}")
    print(f"Probability for Option C: {probability_C:.2f}")
    print(f"Probability for Option D: {probability_D:.2f}")

print(weights_A)


#Question 12 Fall 2018