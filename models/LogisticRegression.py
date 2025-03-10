import numpy as np

class LogisticRegression:
    def __init__(self, n_features):
        """
        Logistic Regression model initialization.

        Parameters:
        - n_features: Number of input features.
        """
        self.n_features = n_features
        self.W = np.random.randn(1, n_features) * 0.01 # small random weights to break symmetry (e.g, if all weights are the same and updated symmetrically, causes ineffective learning.)
        self.b = 0

    @staticmethod
    def sigmoid(z):
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        """
        Forward propagation step.

        Parameters:
        - X: Input features, shape (n_features, m_samples).

        Returns:
        - A: Predicted probabilities, shape (1, m_samples).
        """
        Z = np.dot(self.W, X) + self.b
        A = self.sigmoid(Z)
        return A

    def compute_cost(self, A, Y):
        """
        Compute binary cross-entropy cost.

        Parameters:
        - A: Predicted probabilities, shape (1, m_samples).
        - Y: True labels, shape (1, m_samples).

        Returns:
        - cost: Computed cost.
        """
        m = Y.shape[1]
        cross_entropy_loss = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return np.squeeze(cross_entropy_loss)

    def backward_propagation(self, X, Y, A):
        """
        Backward propagation step to compute gradients.

        Parameters:
        - X: Input features, shape (n_features, m_samples).
        - Y: True labels, shape (1, m_samples).
        - A: Predicted probabilities, shape (1, m_samples).

        Returns:
        - gradients: Dictionary containing gradients dW and db.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1/m) * np.dot(dZ, X.T) 
        db = (1/m) * np.sum(dZ)

        gradients = {"dW": dW, "db": db}
        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        Update parameters using gradient descent.

        Parameters:
        - gradients: Dictionary containing gradients dW and db.
        - learning_rate: Learning rate for updates.
        """
        self.W -= learning_rate * gradients["dW"]
        self.b -= learning_rate * gradients["db"]

    def fit(self, X, Y, learning_rate=0.01, n_iters=1000, verbose=True):
        """
        Fit model parameters.

        Parameters:
        - X: Input features, shape (n_features, m_samples).
        - Y: True labels, shape (1, m_samples).
        - learning_rate: Step size for parameter updates.
        - n_iters: Number of iterations for training.
        - verbose: Print cost every 10 iterations if True.

        Returns:
        - costs: List of cost values per iteration.
        """
        costs = []
        for i in range(n_iters):
            A = self.forward_propagation(X)
            cost = self.compute_cost(A, Y)
            gradients = self.backward_propagation(X, Y, A)
            self.update_parameters(gradients, learning_rate)
            costs.append(cost)

            if verbose and i % 10 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return costs

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels for given data.

        Parameters:
        - X: Input features, shape (n_features, m_samples).
        - threshold: Probability threshold to classify labels.

        Returns:
        - predictions: Binary predictions, shape (1, m_samples).
        """
        A = self.forward_propagation(X)
        predictions = (A > threshold).astype(int)
        return predictions
