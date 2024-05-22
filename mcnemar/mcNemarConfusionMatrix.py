#SPRING 2023 QUESTION 12

from scipy.stats import chi2_contingency

# Define the contingency tables for McNemar's test
# MA vs MB

# I THINK THIS IS A CONFUSION MATRIX. TP, TF , FP, FN. WE HAVE 2

# EDIT NUMBER 1 AND FINAL. CHANGE THESE NUMBERS. 
table_AB = [[416, 42],
            [38, 68]]

# MA vs MC
table_AC = [[68, 38],
            [42, 416]]

def mcnemars_test(table):
    """
    Perform McNemar's test and calculate the difference in accuracy.
    
    Parameters:
    table (list): Contingency table for McNemar's test.
    
    Returns:
    tuple: p-value and difference in accuracy.
    """
    b = table[0][1]
    c = table[1][0]
    
    # McNemar's test statistic
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    
    # p-value from chi-square distribution with 1 degree of freedom
    p_value = chi2_contingency(table, correction=False)[1]
    
    # Difference in accuracy
    theta_hat = (b - c) / (b + c)
    
    return p_value, theta_hat

# Perform McNemar's test for MA vs MB
p_AB, theta_AB = mcnemars_test(table_AB)

# Perform McNemar's test for MA vs MC
p_AC, theta_AC = mcnemars_test(table_AC)

# Print the results
print(f"p_AB: {p_AB}")
print(f"theta_AB: {theta_AB}")
print(f"p_AC: {p_AC}")
print(f"theta_AC: {theta_AC}")

