import toolbox_extended as te
import toolbox_02450 as tb
import numpy as np

# Assuming you have a dataset X (1-dimensional data)
X = np.array([-0.82, 0, 2.5])  # Replace with your actual data

# Define the kernel width (given in the problem)
w = 2.3

# Calculate the density and log density using the Gaussian kernel density function
density, log_density = tb.gausKernelDensity(X, w)

# Output the density and log density for inspection
print("Density:", density)
print("Log Density:", log_density)
