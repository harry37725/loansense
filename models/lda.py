import numpy as np
import pickle

class LDA:
    def __init__(self, n_components=None):
        """
        n_components: number of linear discriminants to keep
        """
        self.n_components = n_components
        self.scalings_ = None
        self.means_ = None

    def fit(self, X, y):
        """
        Fit LDA model to data
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Compute overall mean
        mean_overall = np.mean(X, axis=0)

        # Initialize scatter matrices
        S_W = np.zeros((n_features, n_features))  # within-class scatter
        S_B = np.zeros((n_features, n_features))  # between-class scatter

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.means_ = mean_c

            # Within-class scatter
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))

            # Between-class scatter
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)

        # Solve generalized eigenvalue problem for inv(S_W) * S_B
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(-eigvals.real)
        eigvecs = eigvecs[:, sorted_indices]

        # Select top n_components
        if self.n_components is not None:
            eigvecs = eigvecs[:, :self.n_components]

        self.scalings_ = eigvecs.real

    def predict_proba(self, X):
        """
        Project data into LDA space and return discriminant scores
        """
        return np.dot(X, self.scalings_)

    def predict(self, X, threshold=0.5):
        """
        Classify based on discriminant scores
        """
        scores = self.predict_proba(X)
        # For binary classification, threshold on first discriminant
        if scores.shape[1] == 1:
            return (scores > threshold).astype(int).flatten()
        else:
            # For multi-class, assign to class with max score
            return np.argmax(scores, axis=1)

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