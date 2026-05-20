import numpy as np
import pickle

class BayesianClassifier:
    def __init__(self, hyperparameters=None):
        # hyperparameters can include smoothing (Laplace)
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.class_priors = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes, counts = np.unique(y, return_counts=True)

        self.class_priors = counts / n_samples

        # Use Gaussian (mean/var) instead of binary feature_probs
        self.feature_mean = {}
        self.feature_var  = {}

        for c in self.classes:
            X_c = X[y == c]
            self.feature_mean[c] = X_c.mean(axis=0)
            self.feature_var[c]  = X_c.var(axis=0) + 1e-9
            
    def predict_proba(self, X):
        """
        Returns probability estimates for each class
        """
        probs = []
        for x in X:
            class_probs = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.class_priors[idx])
                # Gaussian log-likelihood
                var  = self.feature_var[c]
                mean = self.feature_mean[c]
                likelihood = np.sum(-0.5 * np.log(2 * np.pi * var) - ((x - mean)**2) / (2 * var))
                class_probs.append(prior + likelihood)
            # Normalize
            exp_probs = np.exp(class_probs - np.max(class_probs))
            probs.append(exp_probs / exp_probs.sum())
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        """
        Returns predicted class labels
        """
        proba = self.predict_proba(X)
        return np.array([self.classes[np.argmax(p)] for p in proba])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)