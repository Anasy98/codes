#Question 11 2023 fall

import numpy as np

# Given probabilities
p_y_low = 0.53
p_y_high = 0.47

# Conditional probabilities p(x1, x3 | y)
p_x1_0_x3_0_given_low = 0.04
p_x1_0_x3_1_given_low = 0.03
p_x1_1_x3_0_given_low = 0.66
p_x1_1_x3_1_given_low = 0.27

p_x1_0_x3_0_given_high = 0.25
p_x1_0_x3_1_given_high = 0.68
p_x1_1_x3_0_given_high = 0.02
p_x1_1_x3_1_given_high = 0.05

# Step 1: Calculate p(x1 = 0 | y)
p_x1_0_given_low = p_x1_0_x3_0_given_low + p_x1_0_x3_1_given_low
p_x1_0_given_high = p_x1_0_x3_0_given_high + p_x1_0_x3_1_given_high

# Step 2: Calculate p(x1 = 0) using the law of total probability
p_x1_0 = (p_x1_0_given_low * p_y_low) + (p_x1_0_given_high * p_y_high)

# Step 3: Apply Bayes' theorem to find p(y = High | x1 = 0)
p_y_high_given_x1_0 = (p_x1_0_given_high * p_y_high) / p_x1_0

# Print the results
print(f"p(x1 = 0 | y = Low) = {p_x1_0_given_low:.4f}")
print(f"p(x1 = 0 | y = High) = {p_x1_0_given_high:.4f}")
print(f"p(x1 = 0) = {p_x1_0:.4f}")
print(f"p(y = High | x1 = 0) = {p_y_high_given_x1_0:.4f}")

# Optional: Function to wrap everything up
def compute_p_y_high_given_x1_0(p_y_low, p_y_high, p_x1_0_x3_0_given_low, p_x1_0_x3_1_given_low, 
                                p_x1_1_x3_0_given_low, p_x1_1_x3_1_given_low,
                                p_x1_0_x3_0_given_high, p_x1_0_x3_1_given_high, 
                                p_x1_1_x3_0_given_high, p_x1_1_x3_1_given_high):
    p_x1_0_given_low = p_x1_0_x3_0_given_low + p_x1_0_x3_1_given_low
    p_x1_0_given_high = p_x1_0_x3_0_given_high + p_x1_0_x3_1_given_high
    p_x1_0 = (p_x1_0_given_low * p_y_low) + (p_x1_0_given_high * p_y_high)
    p_y_high_given_x1_0 = (p_x1_0_given_high * p_y_high) / p_x1_0
    return p_y_high_given_x1_0

# Call the function and print the result
p_y_high_given_x1_0 = compute_p_y_high_given_x1_0(
    p_y_low, p_y_high, p_x1_0_x3_0_given_low, p_x1_0_x3_1_given_low, 
    p_x1_1_x3_0_given_low, p_x1_1_x3_1_given_low,
    p_x1_0_x3_0_given_high, p_x1_0_x3_1_given_high, 
    p_x1_1_x3_0_given_high, p_x1_1_x3_1_given_high
)

print(f"Computed p(y = High | x1 = 0) = {p_y_high_given_x1_0:.4f}")
