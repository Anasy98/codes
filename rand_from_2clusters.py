#Question8 Spring 2023

from itertools import combinations

def calculate_jaccard_index(Z, Q):
    def count_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            for pair in combinations(cluster, 2):
                pairs.add(frozenset(pair))
        return pairs

    Z_pairs = count_pairs(Z)
    Q_pairs = count_pairs(Q)

    intersection = len(Z_pairs & Q_pairs)
    union = len(Z_pairs | Q_pairs)

    if union == 0:
        return 1
    JI = intersection / union

    return JI

def calculate_rand_index(Z, Q):
    def count_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            for pair in combinations(cluster, 2):
                pairs.add(frozenset(pair))
        return pairs

    Z_pairs = count_pairs(Z)
    Q_pairs = count_pairs(Q)
    
    TP = len(Z_pairs & Q_pairs)
    FP = len(Q_pairs - Z_pairs)
    FN = len(Z_pairs - Q_pairs)
    TN = len(set(combinations(range(1, max(max(Z, key=max)) + 1), 2))) - (TP + FP + FN)
    
    RI = (TP + TN) / (TP + FP + FN + TN)
    return RI

# Example clusterings
Z = [
  
        {1, 2, 3, 4},  # Class C1 (Machine)
    {5,  6, 7, 8, 9 ,10}  # Class C2 (Natural)
 

 # Class 2
]

Q = [

    {1,2,3,4,5,6,7,8,9,10},  # Single cluster
]

# Calculate Jaccard Index
jaccard_index = calculate_jaccard_index(Z, Q)
print(f"The Jaccard Index is: {jaccard_index:.3f}")

# Calculate Rand Index
rand_index = calculate_rand_index(Z, Q)
print(f"The Rand Index is: {rand_index:.3f}")
