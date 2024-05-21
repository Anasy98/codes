#Question 13 Fall 2023

import numpy as np

# Function to compute the multivariate normal probability density
def multivariate_normal_pdf(x, mean, cov):
    k = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_factor = 1 / ((2 * np.pi) ** (k / 2) * (det_cov ** 0.5))
    exp_factor = np.exp(-0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)))
    return norm_factor * exp_factor

# Function to classify the test point using Naive Bayes approach
def classify_naive_bayes(x_test, means, covs, priors):
    posteriors = []
    
    for i in range(len(means)):
        mean = means[i]
        cov = np.diag(np.diag(covs[i]))  # Use diagonal elements only for Naive Bayes
        prior = priors[i]
        
        likelihood = multivariate_normal_pdf(x_test, mean, cov)
        posterior = likelihood * prior
        posteriors.append(posterior)
    
    return np.argmax(posteriors), posteriors

# Main function to handle the entire process
def main():
    # Given data (Example dataset)
    means = [
        np.array([0.77, -0.41]),  # Mean for class 1 (Low)
        np.array([-0.91, 0.5])    # Mean for class 2 (High)
    ]
    
    covs = [
        np.array([[0.29, -0.12], [-0.12, 0.55]]),  # Covariance matrix for class 1 (Low)
        np.array([[0.32, -0.11], [-0.11, 1.12]])   # Covariance matrix for class 2 (High)
    ]
    
    priors = [0.53, 0.47]  # Prior probabilities for class 1 (Low) and class 2 (High)
    
    x_test = np.array([0, 0.7])  # Test point to classify
    
    # Classify the test point
    class_label, posteriors = classify_naive_bayes(x_test, means, covs, priors)
    
    # Print the results
    print(f"Posterior probabilities: {posteriors}")
    print(f"Classified as class: {class_label + 1}")  # Adding 1 to match class labels 1 and 2

# Run the main function
if __name__ == "__main__":
    main()
