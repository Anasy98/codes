# FALL 2020 QUESTION 18

def calculate_classification_error(probabilities):
    return 1 - max(probabilities)

def calculate_purity_gain(N_parent, N_left, N_right, classError_parent, classError_left, classError_right):
    weighted_class_error = (N_left / N_parent) * classError_left + (N_right / N_parent) * classError_right
    purity_gain = classError_parent - weighted_class_error
    return purity_gain

# Given data
N_parent = 333
N_left = 265
N_right = 68

# Probabilities in the parent node
p_adelie_parent = 146 / N_parent
p_gentoo_parent = 119 / N_parent
p_chinstrap_parent = 68 / N_parent

# Calculate classification error for the parent node
classError_parent = calculate_classification_error([p_adelie_parent, p_gentoo_parent, p_chinstrap_parent])

# Probabilities in the left node
p_adelie_left = 146 / N_left
p_gentoo_left = 119 / N_left
p_chinstrap_left = 0

# Calculate classification error for the left node
classError_left = calculate_classification_error([p_adelie_left, p_gentoo_left, p_chinstrap_left])

# Probabilities in the right node
p_adelie_right = 0
p_gentoo_right = 0
p_chinstrap_right = 1

# Calculate classification error for the right node
classError_right = calculate_classification_error([p_adelie_right, p_gentoo_right, p_chinstrap_right])

# Calculate purity gain
purity_gain = calculate_purity_gain(N_parent, N_left, N_right, classError_parent, classError_left, classError_right)
print(f"Purity Gain: {purity_gain:.4f}")
