import numpy as np
import pickle

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, 
                 value=None, prob=None):
        self.feature_index = feature_index
        self.threshold     = threshold
        self.left          = left
        self.right         = right
        self.value         = value   # class label for leaf
        self.prob          = prob    # P(class=1) for leaf

class DecisionTreeClassifier:
    def __init__(self, hyperparameters=None):
        self.hyperparameters   = hyperparameters if hyperparameters else {}
        self.max_depth         = self.hyperparameters.get("max_depth", 5)
        self.min_samples_split = self.hyperparameters.get("min_samples_split", 500)
        self.root              = None

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
        best_gain             = 0
        parent_impurity       = self.gini(y)

        for feature_idx in range(n_features):
            # ✅ Sample 20 percentile thresholds instead of all unique values
            thresholds = np.percentile(X[:, feature_idx], np.linspace(5, 95, 20))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask  = X[:, feature_idx] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_imp  = self.gini(y[left_mask])
                right_imp = self.gini(y[right_mask])
                n_left, n_right = left_mask.sum(), right_mask.sum()
                weighted_imp = (n_left * left_imp + n_right * right_imp) / n_samples
                gain = parent_impurity - weighted_imp

                if gain > best_gain:
                    best_gain   = gain
                    best_idx    = feature_idx
                    best_thresh = thresh

        return best_idx, best_thresh

    def build_tree(self, X, y, depth=0):
        classes, counts = np.unique(y, return_counts=True)
        # ✅ Store P(class=1) at every leaf for proper predict_proba
        prob = np.sum(y == 1) / len(y)

        if len(classes) == 1:
            return DecisionTreeNode(value=classes[0], prob=prob)
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=classes[np.argmax(counts)], prob=prob)

        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            return DecisionTreeNode(value=classes[np.argmax(counts)], prob=prob)

        left_mask  = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left_child  = self.build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(feature_index=feature_idx, threshold=threshold,
                                left=left_child, right=right_child, prob=prob)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict_proba_sample(self, x, node):
        if node.value is not None:
            return node.prob       # ✅ real probability, not one-hot
        if x[node.feature_index] <= node.threshold:
            return self.predict_proba_sample(x, node.left)
        else:
            return self.predict_proba_sample(x, node.right)

    def predict(self, X, threshold=0.5):
        return np.array([self.predict_sample(x, self.root) for x in X])

    def predict_proba(self, X):
        # ✅ Returns 1-D array of P(class=1) — works with extract_proba in train.py
        return np.array([self.predict_proba_sample(x, self.root) for x in X])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)