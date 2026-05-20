import numpy as np
import pickle

class NeuralNetworkClassifier:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.input_dim = self.hyperparameters.get("input_dim", None)
        self.hidden_dim = self.hyperparameters.get("hidden_dim", 16)
        self.output_dim = self.hyperparameters.get("output_dim", 2)
        self.learning_rate = self.hyperparameters.get("learning_rate", 0.01)
        self.n_epochs = self.hyperparameters.get("n_epochs", 1000)

        # Initialize weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.01
        self.b2 = np.zeros((1, self.output_dim))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        y_onehot = np.eye(self.output_dim)[y]

        for epoch in range(self.n_epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.softmax(z2)

            # Backpropagation
            dz2 = a2 - y_onehot
            dW2 = np.dot(a1.T, dz2) / X.shape[0]
            db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

            dz1 = np.dot(dz2, self.W2.T) * a1 * (1 - a1)
            dW1 = np.dot(X.T, dz1) / X.shape[0]
            db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict_proba(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self.softmax(z2)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)