#spring 2019. question 19.  replace the data with your information and you should be good to go. 

# this is solid. tested out and worked in 2 scenarios 100% accuracy. tested on spring 2018 question 6 tho
# and got mixed signals. so 2out 3. be carefule about questions wher it is empty. 

# this is only for backward and forward selection questions. 

import pandas as pd

# Example data
data = {
    "Feature(s)": [
        "none", "x2", "x3", "x4", "x5", "x6", "x2, x3", "x2, x4", "x2, x5", "x2, x6",
        "x3, x4", "x3, x5", "x3, x6", "x4, x5", "x4, x6", "x5, x6", "x2, x3, x4",
        "x2, x3, x5", "x2, x3, x6", "x2, x4, x5", "x2, x4, x6", "x2, x5, x6",
        "x3, x4, x5", "x3, x4, x6", "x3, x5, x6", "x4, x5, x6", "x2, x3, x4, x5",
        "x2, x3, x4, x6", "x2, x3, x5, x6", "x2, x4, x5, x6", "x3, x4, x5, x6",
        "x2, x3, x4, x5, x6"
    ],
    "Training RMSE": [
        0.11279, 0.10930, 0.10974, 0.10911, 0.11254, 0.09301, 0.10914, 0.10756,
        0.10909, 0.09108, 0.10837, 0.10961, 0.09108, 0.10910, 0.09234, 0.08993,
        0.10753, 0.10887, 0.09071, 0.10731, 0.08947, 0.08900, 0.10828, 0.08805,
        0.08896, 0.08891, 0.10730, 0.08782, 0.08878, 0.08727, 0.08603, 0.08595
    ],
    "Test RMSE": [
        0.20677, 0.22301, 0.21773, 0.21362, 0.20729, 0.18749, 0.22247, 0.22145,
        0.22513, 0.18555, 0.21768, 0.21800, 0.17624, 0.21368, 0.19121, 0.17657,
        0.22138, 0.22435, 0.18029, 0.22315, 0.19339, 0.17610, 0.21795, 0.17900,
        0.17082, 0.18062, 0.22314, 0.18371, 0.17299, 0.18336, 0.17440, 0.17685
    ]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

def forward_selection(df):
    """
    Perform forward selection to select the best features based on Test RMSE.

    Parameters:
    df (DataFrame): DataFrame containing features and their corresponding RMSE values.

    Returns:
    List of tuples: Each tuple contains the selected feature set and its Test RMSE.
    """
    remaining_features = set(df['Feature(s)']) - {"none"}
    selected_features = []
    current_score, best_new_score = float("inf"), float("inf")
    best_features = []

    while remaining_features and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining_features:
            features_to_evaluate = selected_features + [candidate]
            feature_str = "; ".join(sorted(features_to_evaluate))
            if feature_str in df['Feature(s)'].values:
                score = df.loc[df['Feature(s)'] == feature_str, 'Test RMSE'].values[0]
                scores_with_candidates.append((score, candidate))
            else:
                print(f"Feature combination '{feature_str}' not found in DataFrame.")
        
        if not scores_with_candidates:
            break

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if current_score == float("inf") or best_new_score < current_score:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_new_score
            best_features.append((selected_features[:], current_score))

    return best_features

def backward_selection(df):
    """
    Perform backward selection to select the best features based on Test RMSE.

    Parameters:
    df (DataFrame): DataFrame containing features and their corresponding RMSE values.

    Returns:
    List of tuples: Each tuple contains the selected feature set and its Test RMSE.
    """
    initial_features = set(df['Feature(s)'].max().split("; "))
    selected_features = list(initial_features)
    current_score, best_new_score = float("inf"), float("inf")
    best_features = []

    while selected_features and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in selected_features:
            features_to_evaluate = list(set(selected_features) - {candidate})
            feature_str = "; ".join(sorted(features_to_evaluate)) if features_to_evaluate else "none"
            if feature_str in df['Feature(s)'].values:
                score = df.loc[df['Feature(s)'] == feature_str, 'Test RMSE'].values[0]
                scores_with_candidates.append((score, candidate))
            else:
                print(f"Feature combination '{feature_str}' not found in DataFrame.")
        
        if not scores_with_candidates:
            break

        scores_with_candidates.sort()
        best_new_score, worst_candidate = scores_with_candidates[0]

        if current_score == float("inf") or best_new_score < current_score:
            selected_features.remove(worst_candidate)
            current_score = best_new_score
            best_features.append((selected_features[:], current_score))

    return best_features

# Running forward selection
forward_selected_features = forward_selection(df)
print("Forward Selection Results:")
for features, score in forward_selected_features:
    print(f"Features: {features}, Test RMSE: {score}")

# Running backward selection
backward_selected_features = backward_selection(df)
print("Backward Selection Results:")
for features, score in backward_selected_features:
    print(f"Features: {features}, Test RMSE: {score}")
