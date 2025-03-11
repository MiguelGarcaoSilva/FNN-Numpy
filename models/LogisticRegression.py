import numpy as np

class LogisticRegressionBinaryClassifier:
    def __init__(self, n_inputs):
        """
        Logistic Regression model initialization.

        Parameters:
        - n_inputs: Number of input features.
        """
        self.n_inputs = n_inputs
        self.W = np.random.randn(1, n_inputs) * 0.01 # small random weights to break symmetry (e.g, if all weights are the same and updated symmetrically, causes ineffective learning.)
        self.b = 0

    @staticmethod
    def sigmoid(z):
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        """
        Forward propagation step.

        Parameters:
        - X: Input features, shape (n_inputs, m).

        Returns:
        - A: Predicted probabilities, shape (1, m).
        """
        Z = np.dot(self.W, X) + self.b
        A = self.sigmoid(Z)
        return A

    def compute_cost(self, A, Y):
        """
        Compute binary cross-entropy cost.

        Parameters:
        - A: Predicted probabilities, shape (1, m).
        - Y: True labels, shape (1, m).

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
        - X: Input features, shape (n_inputs, m).
        - Y: True labels, shape (1, m).
        - A: Predicted probabilities, shape (1, m).

        Returns:
        - gradients: Dictionary containing gradients dW and db.
        """
        m = Y.shape[1]
        dZ = A - Y # Gradient of cost w.r.t linear output (cross-entropy + sigmoid/softmax simplification)
        dW = (1/m) * np.dot(dZ, X.T)  # gradient w.r.t weights W
        db = (1/m) * np.sum(dZ) # gradient w.r.t bias 

        return {"dW": dW, "db": db}

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
        - X: Input features, shape (n_inputs, m).
        - Y: True labels, shape (1, m).
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

    def predict(self, X):
        """
        Predict binary labels for given data.

        Parameters:
        - X: Input features, shape (n_inputs, m).
        
        Returns:
        - predictions: Binary predictions, shape (1, m).
        """
        AL = self.forward_propagation(X)
        return AL > 0.5
