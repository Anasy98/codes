#Question 23 Spring 2018

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples  # Initialize weights

        for _ in range(self.n_estimators):
            model = LogisticRegression(solver='lbfgs')
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            error = np.sum(w * (y_pred != y)) / np.sum(w)
            alpha = 0.5 * np.log((1 - error) / error)

            w = w * np.exp(-alpha * y * y_pred)
            w = w / np.sum(w)  # Normalize to make it a probability distribution

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])

        for model, alpha in zip(self.models, self.alphas):
            predictions = model.predict(X)
            final_predictions += alpha * predictions

        return np.sign(final_predictions)

# Function to calculate and print the accuracy of the AdaBoost classifier
def evaluate_adaboost(X_train, y_train, X_test, y_test, n_estimators=50):
    adaboost = AdaBoost(n_estimators=n_estimators)
    adaboost.fit(X_train, y_train)
    y_pred_train = adaboost.predict(X_train)
    y_pred_test = adaboost.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

def main():
    # === Example dataset ===
    # This section uses a synthetic dataset for demonstration purposes.
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    y = np.where(y == 0, -1, 1)  # Ensure labels are -1 and 1 for AdaBoost

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # === End of example dataset ===

    # Evaluate AdaBoost with Logistic Regression
    evaluate_adaboost(X_train, y_train, X_test, y_test, n_estimators=10)

if __name__ == "__main__":
    main()
