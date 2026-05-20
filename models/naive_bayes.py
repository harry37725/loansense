import numpy as np
import pickle

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Fit Naive Bayes model
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-9   # prevent division by zero
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.clip(numerator / denominator, 1e-300, None)

    def predict_proba(self, X):
        """
        Return class probabilities for each sample
        """
        probs = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._pdf(c, x)))
                class_probs[c] = prior + likelihood
            # normalize to probabilities
            max_log = max(class_probs.values())
            exp_scores = {c: np.exp(v - max_log) for c, v in class_probs.items()}
            total = sum(exp_scores.values())
            probs.append({c: exp_scores[c] / total for c in self.classes})
        return probs

    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        """
        probs = self.predict_proba(X)
        predictions = []
        for p in probs:
            # pick class with max probability
            predictions.append(max(p, key=p.get))
        return np.array(predictions)

    def save(self, path):
        """
        Save model to file
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        """
        Load model from file
        """
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f)
