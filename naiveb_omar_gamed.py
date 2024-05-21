import math
from typing import Optional, Dict, Any, Union

class NaiveBayesClassifier:
    def __init__(self):
        self.parameters = {
            'number_of_classes': None,
            'number_of_features': None,
            'class_probs': {},  # e.g., P(y=1), P(y=2), ...
            'feature_probs_given_class': {},  # e.g., P(f1=0|y=1), P(f2=1|y=2), ...
            'joint_feature_probs_given_class': {},  # e.g., P(f1=0, f6=1|C2)
            'mean_given_class': {},  # e.g., mean_x1_given_y1, mean_x2_given_y2, ...
            'var_given_class': None,  # variance for normal density
            'x_values': {},  # e.g., {'x1': 32.0, 'x2': 14.0}
            'class_labels': {},  # e.g., {'C1': 1, 'C2': 2, 'C3': 3}
            'class_label_probs': {}  # e.g., {'C1': 0.18, 'C2': 0.45, 'C3': 0.36}
        }

    def set_parameter(self, key: str, value: Any):
        if key in ['number_of_classes', 'number_of_features', 'var_given_class']:
            self.parameters[key] = value
        elif key.startswith('P_y'):
            class_label = int(key.split('_')[-1][1:])
            self.parameters['class_probs'][class_label] = value
        elif key.startswith('P_f'):
            parts = key.split('_')
            feature = parts[1]
            condition = int(parts[-1][1:])
            if condition not in self.parameters['feature_probs_given_class']:
                self.parameters['feature_probs_given_class'][condition] = {}
            self.parameters['feature_probs_given_class'][condition][feature] = value
        elif key.startswith('P_joint'):
            parts = key.split('_')
            feature_combo = '_'.join(parts[1:-1])
            condition = parts[-1]
            if condition not in self.parameters['joint_feature_probs_given_class']:
                self.parameters['joint_feature_probs_given_class'][condition] = {}
            self.parameters['joint_feature_probs_given_class'][condition][feature_combo] = value
        elif key.startswith('mean'):
            parts = key.split('_')
            feature = parts[1]
            condition = int(parts[-1][1:])
            if condition not in self.parameters['mean_given_class']:
                self.parameters['mean_given_class'][condition] = {}
            self.parameters['mean_given_class'][condition][feature] = value
        elif key.startswith('x'):
            self.parameters['x_values'][key] = value
        elif key.startswith('C'):
            self.parameters['class_label_probs'][key] = value
            self.parameters['class_labels'][key] = len(self.parameters['class_labels']) + 1
        else:
            print(f"Invalid key: {key}")

    def get_joint_prob(self, feature_combo: str, class_label: str) -> float:
        """
        Calculate the joint probability P(feature_combo | class_label)
        """
        if class_label in self.parameters['joint_feature_probs_given_class']:
            return self.parameters['joint_feature_probs_given_class'][class_label].get(feature_combo, 0)
        return 0

    def get_naive_bayes_prob(self, y: int) -> float:
        """
        Calculate the Naive Bayes probability for a given class y (0, 1, 2, ...)
        """
        P_y = self.parameters['class_probs'][y]
        prob = P_y
        
        for feature, value in self.parameters['x_values'].items():
            P_x_given_y = self.parameters['feature_probs_given_class'][y].get(feature)
            if P_x_given_y is not None:
                prob *= P_x_given_y

        return prob

    def calculate_all_probabilities(self) -> Dict[str, Union[float, str]]:
        """
        Calculate all possible results based on the provided parameters.
        """
        results = {}

        # Print debug information
        print("Class probabilities:", self.parameters['class_probs'])
        print("Feature probabilities given class:", self.parameters['feature_probs_given_class'])
        print("Joint feature probabilities given class:", self.parameters['joint_feature_probs_given_class'])
        print("X values:", self.parameters['x_values'])
        print("Class labels:", self.parameters['class_labels'])
        print("Class label probabilities:", self.parameters['class_label_probs'])

        # Calculate joint probabilities for all f_values combinations given class labels
        for class_key, class_label in self.parameters['class_labels'].items():
            if self.parameters['joint_feature_probs_given_class']:
                if class_key in self.parameters['joint_feature_probs_given_class']:
                    for feature_combo, joint_prob in self.parameters['joint_feature_probs_given_class'][class_key].items():
                        print(f"Calculating joint probability P({feature_combo}|{class_key})")
                        results[f'P({feature_combo}|{class_key})'] = joint_prob
                else:
                    print(f"Warning: No joint feature probabilities found for class {class_key} (label {class_label})")

        # Calculate P(Ci|feature_combo) for all classes using Bayes' theorem
        for feature_combo in set([combo for class_probs in self.parameters['joint_feature_probs_given_class'].values() for combo in class_probs]):
            p_f = sum(self.get_joint_prob(feature_combo, key) * self.parameters['class_label_probs'][key] for key in self.parameters['class_labels'])
            print(f"Total probability P({feature_combo}) = {p_f:.4f}")
            for class_key in self.parameters['class_labels']:
                if feature_combo in self.parameters['joint_feature_probs_given_class'].get(class_key, {}):
                    p_f_given_c = self.get_joint_prob(feature_combo, class_key)
                    p_c = self.parameters['class_label_probs'][class_key]
                    if p_f > 0:
                        results[f'P({class_key}|{feature_combo})'] = (p_f_given_c * p_c) / p_f
                        print(f"Calculated P({class_key}|{feature_combo}) = {(p_f_given_c * p_c) / p_f:.4f}")

        # If Gaussian densities are provided, calculate P(feature|y) for all features and classes
        if self.parameters['mean_given_class'] and self.parameters['var_given_class'] is not None:
            for feature, x in self.parameters['x_values'].items():
                for y in self.parameters['mean_given_class']:
                    if feature in self.parameters['mean_given_class'][y]:
                        mean_y = self.parameters['mean_given_class'][y][feature]
                        var = self.parameters['var_given_class']
                        p_x_given_y = self.gaussian_density(x, mean_y, var)
                        results[f'P({feature}={x}|y={y})'] = p_x_given_y
                        print(f"Calculated Gaussian probability P({feature}={x}|y={y}) = {p_x_given_y:.4f}")

        # Normalize and calculate final probabilities for each class given x
        total_prob = sum(results.values())
        for key in list(results.keys()):
            if 'normalized' not in key:
                results[f'{key} (normalized)'] = results[key] / total_prob if total_prob > 0 else 0
                print(f"Normalized probability {key} = {results[f'{key} (normalized)']:.4f}")

        return results

    @staticmethod
    def gaussian_density(x: float, mean: float, var: float) -> float:
        return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean) ** 2) / (2 * var))


# Example Usage:
nb = NaiveBayesClassifier()

# Set parameter
nb.set_parameter('number_of_classes', 2)
nb.set_parameter('number_of_features', 10)
# nb.set_parameter('P_y1', 0.00027)
# nb.set_parameter('P_y2', 0.000004)
# nb.set_parameter('P_y3', 0.000003)

nb.set_parameter('P_joint_x2=1_x4=0_C1', 0.18/0.53)
nb.set_parameter('P_joint_x2=1_x4=0_C2', 0.32/0.47)
# nb.set_parameter('P_joint_x2=1_x10=0_C3', 0.35)

# nb.set_parameter('mean_x1_given_y1', 32.4)
# nb.set_parameter('mean_x2_given_y1', 13.95)
# nb.set_parameter('var_given_class', 400)
# nb.set_parameter('x1', 32.0)
# nb.set_parameter('x2', 14.0)
nb.set_parameter('C1', 0.53)
nb.set_parameter('C2', 0.47)
# nb.set_parameter('C3', 0.328)



# Calculate results
results = nb.calculate_all_probabilities()

print('hello1')

# Print results
for key, value in results.items():
    print(f"{key}: {value:.4f}")

print('hello2')
