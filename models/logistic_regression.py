import numpy as np

class LogisticRegression:
    """
    Logistic Regression from scratch using MLE + Gradient Descent
    
    P(y=1|x) = sigmoid(w^T x + b)
    Loss     = -1/N * sum[ y*log(p) + (1-y)*log(1-p) ]  (Binary Cross Entropy = MLE)
    Update   = w = w - lr * dL/dw
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-4):
        self.lr           = learning_rate
        self.n_iterations = n_iterations
        self.tolerance    = tolerance
        self.weights      = None
        self.bias         = None
        self.loss_history = []

    # ------------------------------------------------------------------ #
    #  Private Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _sigmoid(self, z):
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        # Binary cross entropy (MLE derivation)
        N = len(y)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # avoid log(0)
        return -1/N * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _compute_gradients(self, X, y, y_pred):
        N = len(y)
        error = y_pred - y               # (N,)
        dw = (1/N) * X.T @ error         # (features,)
        db = (1/N) * np.sum(error)       # scalar
        return dw, db

    # ------------------------------------------------------------------ #
    #  Public Interface                                                    #
    # ------------------------------------------------------------------ #

    def fit(self, X, y, verbose=True):
        """
        Train model using gradient descent
        X : (N, features)
        y : (N,)  binary 0/1
        """
        N, n_features = X.shape
        
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):

            # Forward pass
            z      = X @ self.weights + self.bias   # linear combination
            y_pred = self._sigmoid(z)               # probabilities

            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward pass
            dw, db = self._compute_gradients(X, y, y_pred)

            # Update weights
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            # Print every 100 iterations
            if verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Loss: {loss:.4f}")

            # Early stopping if loss barely changes
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tolerance:
                print(f"Converged at iteration {i}")
                break

        return self

    def predict_proba(self, X):
        """ Returns probability of default (class 1) """
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """ Returns binary prediction 0 or 1 """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        """ Save weights to disk """
        np.save(path, {
            'weights'  : self.weights,
            'bias'     : self.bias,
            'loss_history': self.loss_history
        })
        print(f"✅ Model saved to {path}")

    def load(self, path):
        """ Load weights from disk """
        data = np.load(path, allow_pickle=True).item()
        self.weights      = data['weights']
        self.bias         = data['bias']
        self.loss_history = data['loss_history']
        print(f"✅ Model loaded from {path}")