import numpy as np

# Remember to keep the 1 in the x input vector for the intercept
x = np.array([1, 6, 120, 3.2, 0, 4])

# Weights for the hidden units
w1 = np.array([-4, 1, 0.01, 1, -1, -1])
w2 = np.array([-10, 1, -0.02, 1, 1, 1])

# Weights for the output layer
w2_0 = 7
w2_1 = 8
w2_2 = 9

def tanh(z):
    return np.tanh(z)

# Calculate the hidden layer activations
z1 = np.dot(x, w1)
z2 = np.dot(x, w2)
h1 = tanh(z1)
h2 = tanh(z2)

# Calculate the output
output = w2_1 * h1 + w2_2 * h2 + w2_0
print(output)
