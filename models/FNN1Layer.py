import numpy as np

class FNN1Layer:
    def __init__(self, input_dim, nunits=2, dropout=None):
        """
        Initialize neural network parameters.

        Parameters:
            input_dim (int): Size of the input layer.
            output_dim (int): Size of the output layer.
            nunits (int): Number of hidden units.
            dropout (float): Dropout parameter to regularize the network.
        """
        self.input_dim = input_dim
        self.output_dim = 1
        self.nunits = nunits
        self.dropout = dropout # prevents overfitting by keeping the network from relying too much on any one unit, multiple sub-networks are trained
        self.parameters = {}

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def initialize_parameters_zeros(self):
        """
        Initialize weights and biases to zeros.
        Note: Using zeros results in symmetric hidden units, reducing the network to a linear model (no learning).
        """
        self.parameters = {
            'W1': np.zeros((self.nunits, self.input_dim)),
            'b1': np.zeros((self.nunits, 1)),
            'W2': np.zeros((self.output_dim, self.nunits)),
            'b2': np.zeros((self.output_dim, 1)),
        }

    def initialize_parameters_random(self):
        """Random initialization to break symmetry."""
        self.parameters = {
            'W1': np.random.randn(self.nunits, self.input_dim) * 0.01,
            'b1': np.zeros((self.nunits, 1)),
            'W2': np.random.randn(self.output_dim, self.nunits) * 0.01,
            'b2': np.zeros((self.output_dim, 1)),
        }

    def initialize_parameters_he(self):
        """ He initialization, similar to Xavier but suited for layers with ReLU activations."""
        self.parameters['W1'] = np.random.randn(self.nunits, self.input_dim) * np.sqrt(2 / self.input_dim)
        self.parameters['b1'] = np.zeros((self.nunits, 1))
        self.parameters['W2'] = np.random.randn(self.output_dim, self.nunits) * np.sqrt(2 / self.nunits)
        self.parameters['b2'] = np.zeros((self.output_dim, 1))

    def forward_propagation(self, X):
        """Perform forward propagation.
        
        Parameters:
            X (np.ndarray): Input data of shape (input_dim, m).
            
        Returns:
            AL (np.ndarray): Output layer activation.
            cache (dict): Dictionary containing intermediate values.
        """
        W1, b1, W2, b2 = (self.parameters['W1'], self.parameters['b1'],
                          self.parameters['W2'], self.parameters['b2'])

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)

        if self.dropout:
            D1 = np.random.rand(A1.shape[0], A1.shape[1]) < 1 - self.dropout # dropout mask
            A1 = np.multiply(A1, D1)
            A1 /= 1 - self.dropout # inverted dropout to keep the expected value of the activations the same

        Z2 = np.dot(W2, A1) + b2
        AL = self.sigmoid(Z2)

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'AL': AL}

        if self.dropout: cache['D1'] = D1

        return AL, cache

    def compute_cost(self, AL, Y):
        """Compute binary cross-entropy cost.
        
        Parameters:
            AL (np.ndarray): Output layer activation.
            Y (np.ndarray): True labels of shape (1, m).
        
        Returns:
            cost (float): Computed cost.
         """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return np.squeeze(cost)

    def backward_propagation(self, cache, X, Y):
        """Perform backward propagation.
        
        Parameters:
            cache (dict): Dictionary containing intermediate values.
            X (np.ndarray): Input data of shape (input_dim, m).
            Y (np.ndarray): True labels of shape (1, m).
            
        Returns:
            grads (dict): Dictionary containing gradients.
        """
        m = X.shape[1]
        W2 = self.parameters['W2']
        A1, AL = cache['A1'], cache['AL']

        dZ2 = AL - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(W2.T, dZ2)
        if self.dropout:
            D1 = cache['D1']
            dA1 = np.multiply(dA1, D1) / (1 - self.dropout)

        dZ1 = dA1 * (1 - np.square(A1)) # tanh derivative
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

    def update_parameters(self, grads, learning_rate):
        """Update parameters using gradient descent."""
        for param in self.parameters:
            self.parameters[param] -= learning_rate * grads[f'd{param}']

    def fit(self, X, Y, learning_rate=0.01, n_iters=1000, verbose=True):
        """Train the neural network."""
        np.random.seed(0)
        self.initialize_parameters_he()
        costs = []

        for i in range(n_iters):
            AL, cache = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(cache, X, Y)
            self.update_parameters(grads, learning_rate)

            costs.append(cost)
            if verbose and i % 10 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return costs

    def predict(self, X):
        """Predict output using trained model."""
        AL, _ = self.forward_propagation(X)
        return AL > 0.5
