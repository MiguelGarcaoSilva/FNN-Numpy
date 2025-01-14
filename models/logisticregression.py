import numpy as np

class LogisticRegression:
    def __init__(self, n_inputs, lambd_reg=0.00):
        self.n_inputs = n_inputs
        self.parameters = {}
        self.lambd_reg = lambd_reg

    @staticmethod
    def sigmoid(x):
        '''Sigmoid function that works with numpy arrays'''
        return 1 / (1 + np.exp(-x))

    def initialize_parameters(self):
        '''Initialize weights and bias randomly
        with small random numbers to break symmetry and avoid zero gradients
        '''
        parameters = {}
        parameters['W'] = np.random.randn(1, self.n_inputs) * np.sqrt(1 / self.n_inputs) # xavier initialization
        parameters['b'] = np.zeros((1, 1))
        self.parameters = parameters

    def forward_propagation(self, X):
        m = X.shape[1]
        # Forward pass, sigmoid activation to get the probability
        # (1, m) = (1, n) . (n, m)  + (1, 1) broadcasted
        Z = np.dot(self.parameters["W"], X) + self.parameters["b"]
        A = self.sigmoid(Z)
        return A

    def compute_cost(self, A, y):
        '''Compute the cost -1/m * sum(y * log(A) + (1-y) * log(1-A)) - binary cross-entropy loss'''
        m = y.shape[1]
        # L2 reg ||w||^2 = sum(w^2) = w1^2 + w2^2 + ... + wn^2 = w^T . w
        # y has shape (1, m), A has shape (1, m)
        cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) + self.lambd_reg / (2 * m) * np.sum(np.square(self.parameters["W"]))
        cost = np.squeeze(cost)
        return cost


    def backward_propagation(self, X, y, A):
        grads = {}
        m = y.shape[1]
        # Backward propagation, compute the gradients
        # dw = 1/m \sum (A - y) * X
        # db = 1/m \sum (A - y)
        grads["dW"] = 1 / m * np.dot(A - y, X.T) + self.lambd_reg / m * self.parameters["W"]
        grads["db"] = 1 / m * np.sum(A - y)
        
        return grads

    def update_parameters(self, dw, db, lr):
        '''Update the weights and bias'''
        self.parameters["W"] -= lr * dw
        self.parameters["b"] -= lr * db

    
    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(0)
        costs = []
        self.initialize_parameters()

        for i in range(n_iters):
            A = self.forward_propagation(X)
            cost = self.compute_cost(A, y)
            grads = self.backward_propagation(X, y, A)
            self.update_parameters(grads["dW"], grads["db"], learning_rate)

            # Print the cost every 100 iterations
            if i % 100 == 0:
                costs.append(cost)
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters

    def predict(self, X):
        '''Predict the class labels for the provided data'''
        A = self.forward_propagation(X)
        return A > 0.5