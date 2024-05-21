#Question 27 Fall 2020

import numpy as np

def generate_cross_validation_counts(N, outer_folds, lambdas):
    """
    Simulate two-level cross-validation process and count how many times an observation is used for training.
    
    Parameters:
    N (int): Number of observations
    outer_folds (int): Number of outer folds
    lambdas (list): List of different regularization constants
    
    Returns:
    int: Number of times an observation is used for training
    """
    # Calculate the number of observations in each outer fold
    observations_per_outer_fold = N // outer_folds
    
    # Calculate the number of training sets in inner fold
    inner_training_sets = observations_per_outer_fold - 1
    
    # Calculate the number of models trained per inner training set
    models_per_inner_training_set = len(lambdas)
    
    # Calculate the total models trained per outer training set
    total_models_per_outer_training_set = inner_training_sets * models_per_inner_training_set + 1
    
    # Since each observation is used in (outer_folds - 1) outer folds
    total_training_instances = (outer_folds - 1) * total_models_per_outer_training_set
    
    return total_training_instances

def main():
    # === Insert your dataset parameters here ===
    N = 333  # Total number of observations
    outer_folds = 3  # Number of outer folds
    
    # === Insert your list of regularization constants here ===
    lambdas = [0.001, 0.01, 0.1, 1.0]  # List of different regularization constants

    # Calculate the number of times an observation is used for training
    training_count = generate_cross_validation_counts(N, outer_folds, lambdas)
    
    # Print the result
    print(f"An observation is used {training_count} times for training in this two-level cross-validation setup.")

if __name__ == "__main__":
    main()
