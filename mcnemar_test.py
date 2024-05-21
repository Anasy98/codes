#Question 16 fall 2023

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(data):
    """
    Perform the McNemar test on a given dataset.
    
    Parameters:
    data (2x2 array): Contingency table of the form:
                      [[n_00, n_01],
                       [n_10, n_11]]
    
    Returns:
    result: Result of the McNemar test including the test statistic and p-value.
    """
    # Perform the McNemar test
    result = mcnemar(data, exact=False, correction=True)
    return result

def main():
    # Example dataset
    # Contingency table of the form:
    # [[n_00, n_01],
    #  [n_10, n_11]]
    data_m1_vs_m2 = np.array([[2, 22],  # Replace these values with your dataset
                              [18, 8]])
    
    data_m1_vs_m3 = np.array([[10, 14],  # Replace these values with your dataset
                              [10, 1]])
    
    # Perform McNemar test for M1 vs M2
    result_m1_vs_m2 = mcnemar_test(data_m1_vs_m2)
    print("McNemar Test M1 vs M2")
    print("Statistic: ", result_m1_vs_m2.statistic)
    print("p-value: ", result_m1_vs_m2.pvalue)
    print()
    
    # Perform McNemar test for M1 vs M3
    result_m1_vs_m3 = mcnemar_test(data_m1_vs_m3)
    print("McNemar Test M1 vs M3")
    print("Statistic: ", result_m1_vs_m3.statistic)
    print("p-value: ", result_m1_vs_m3.pvalue)

if __name__ == "__main__":
    main()
