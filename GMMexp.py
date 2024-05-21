import numpy as np

# Constants
sigma_squared = 0.25
M = 10
normalization_factor = np.sqrt((2 * np.pi * sigma_squared) ** M)

# Distances from o3 to o7, o8, o9
distances = np.array([2.11, 1.15, 1.09])
squared_distances = distances ** 2

# Gaussian densities calculation for each component
densities = (1 / normalization_factor) * np.exp(-squared_distances / (2 * sigma_squared))

# Sum and average the densities (since the GMM components are equally weighted)
p_o3 = np.sum(densities) / 3
print("Density at o3:", p_o3)



#Fall 2018 Question5