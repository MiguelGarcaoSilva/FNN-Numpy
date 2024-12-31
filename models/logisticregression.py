import numpy as np

class LogisticRegression:
    def __init__(self, n_inputs, learning_rate=0.01, n_iters=1000):
        self.n_inputs = n_inputs
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x):
        '''Sigmoid function that works with numpy arrays'''
        return 1 / (1 + np.exp(-x))

    def initialize_parameters(self):
        '''Initialize weights and bias to zeros'''
        self.weights = np.random.rand(1, self.n_inputs) * 0.01
        self.bias = np.zeros((1, 1))

    def propagate(self, X, y):
        '''Compute the cost and gradients z = w1*x1 + w2*x2 + wn*xn + b = w^T x + b'''

        m = X.shape[1]
        # Forward pass, sigmoid activation to get the probability
        # (1, m) = (1, n) . (n, m)  + (1, 1) broadcasted
        A = self.sigmoid(np.dot(self.weights, X) + self.bias)
        
        # Compute the cost -1/m * sum(y * log(A) + (1-y) * log(1-A)) - binary cross-entropy loss
        # y has shape (1, m)
        # A has shape (1, m)
        cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        # Backward propagation, compute the gradients
        #dw = 1/m \sum (A - y) * X
        dw = 1 / m * np.dot(A - y, X.T)
        db = 1 / m * np.sum(A - y)

        return cost, dw, db
    
    def fit(self, X, y):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(0)
        costs = []
        self.initialize_parameters()

        for i in range(self.n_iters):
            cost, dw, db = self.propagate(X, y)

            # Update the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print the cost every 100 iterations
            if i % 100 == 0:
                costs.append(cost)
                print(f'Cost after iteration {i}: {cost}')

        return costs

    def predict(self, X):
        '''Predict the class labels for the provided data'''
        A = self.sigmoid(np.dot(self.weights, X) + self.bias)
        return A > 0.5

    