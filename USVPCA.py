#Question 3 2023 fall

import numpy as np

# Step 1: Center the data matrix
def center_data(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return X_centered, mean

# Step 2: Perform SVD to get the matrices U, Sigma, and Vt
def perform_svd(X_centered):
    U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return U, Sigma, Vt

# Step 3: Project an observation onto the first two principal components
def project_observation(x, mean, Vt, num_components=2):
    x_centered = x - mean
    V_reduced = Vt.T[:, :num_components]
    z = np.dot(V_reduced.T, x_centered)
    return z

# Step 4: Reconstruct the observation from the projection
def reconstruct_observation(z, mean, Vt, num_components=2):
    V_reduced = Vt.T[:, :num_components]
    x_reconstructed = np.dot(V_reduced, z) + mean
    return x_reconstructed

# Main function to perform PCA and reconstruction
def pca_reconstruction(X, x_new, num_components=2):
    # Center the data matrix
    X_centered, mean = center_data(X)
    
    # Perform SVD
    U, Sigma, Vt = perform_svd(X_centered)
    
    # Project the new observation
    z = project_observation(x_new, mean, Vt, num_components)
    
    # Reconstruct the new observation
    x_reconstructed = reconstruct_observation(z, mean, Vt, num_components)
    
    return x_reconstructed

# Example usage
if __name__ == "__main__":
    # Given data matrix X
    X = np.array([
        [-0.6, -0.6,  2.5, -0.1],
        [-0.8, -0.3, -1.0,  1.2],
        [-0.7,  0.3, -0.2, -0.1],
        [ 1.4,  1.0,  0.1, -2.8],
        [-0.2,  0.8, -1.2,  0.7]
    ])

    # Given new observation x4
    x_new = np.array([1.4, 1.0, 0.1, -2.8])
    
    # Reconstruct the new observation using the first two principal components
    x_reconstructed = pca_reconstruction(X, x_new, num_components=2)
    
    # Print the reconstructed observation
    print("Reconstructed observation:", x_reconstructed)
