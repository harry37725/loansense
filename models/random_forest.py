import numpy as np
import pickle
from collections import Counter

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.max_depth = self.hyperparameters.get("max_depth", None)
        self.min_samples_split = self.hyperparameters.get("min_samples_split", 2)
        self.root = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        best_idx, best_thresh = None, None
        best_gain = 0
        parent_impurity = self.gini(y)

        for feature_idx in range(n_features):
            thresholds = np.percentile(X[:, feature_idx], np.linspace(5, 95, 20))
            thresholds = np.unique(thresholds)
            for thresh in thresholds:
                left_mask = X[:, feature_idx] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_impurity = self.gini(y[left_mask])
                right_impurity = self.gini(y[right_mask])
                n_left, n_right = left_mask.sum(), right_mask.sum()
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_idx = feature_idx
                    best_thresh = thresh

        return best_idx, best_thresh

    def build_tree(self, X, y, depth=0):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 1:
            return DecisionTreeNode(value=classes[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=classes[np.argmax(counts)])

        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            return DecisionTreeNode(value=classes[np.argmax(counts)])

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionTreeNode(feature_index=feature_idx, threshold=threshold,
                                left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X, threshold=0.5):
        return np.array([self.predict_sample(x, self.root) for x in X])


class RandomForestClassifier:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.n_estimators = self.hyperparameters.get("n_estimators", 10)
        self.max_depth = self.hyperparameters.get("max_depth", None)
        self.min_samples_split = self.hyperparameters.get("min_samples_split", 2)
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeClassifier({
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split
            })
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X):
        # Collect predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        probs = []
        for i in range(X.shape[0]):
            counts = Counter(all_preds[:, i])
            total = sum(counts.values())
            prob = {cls: count / total for cls, count in counts.items()}
            probs.append(prob)
        return probs

    def predict(self, X, threshold=0.5):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            counts = Counter(all_preds[:, i])
            final_preds.append(counts.most_common(1)[0][0])
        return np.array(final_preds)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
