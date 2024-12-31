import numpy as np

class NeuralNetwork1Layer:
    def __init__(self, input_dim , output_dim, nunits=2, learning_rate=0.01, n_iters=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nunits = nunits
        self.lr = learning_rate
        self.n_iters = n_iters
        self.parameters = {}

    @staticmethod
    def sigmoid(x):
        '''Sigmoid function that works with numpy arrays'''
        return 1 / (1 + np.exp(-x))

    def initialize_parameters(self):
        '''Initialize weights and bias randomly
        initializing to zero results in hidden units to be identical'''
        W1 = np.random.randn(self.nunits, self.input_dim) * 0.01 #small random numbers
        b1 = np.zeros((self.nunits, 1))
        W2 = np.random.randn(self.output_dim, self.nunits) * 0.01
        b2 = np.zeros((self.output_dim, 1))
        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

        Forward:
        z^1 = w^1 . X + b^1 ; A^1 = tanh(z^1)
        Z^2 = W^2 . A^1 + b^2 ; A^2 = sigmoid(z^2)
        """      
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        #W1 is shape(nunits, input_dim), 
        #X is shape(input_dim, number of examples), each col is an example
        #b1 is shape(nunits, 1)
        #W2 is shape (1, nunits)

        Z1 =  np.dot(W1, X) + b1 # (nunits,number of examples)
        A1 = np.tanh(Z1) #(nunits,number of examples)
        Z2 = np.dot(W2, A1) + b2 #(1,number of examples)
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}
        return A2, cache

    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost given in equation:
        '''$$J = - \\frac{1}{m} \\sum\\limits_{i = 1}^{m} \\large{(} \\small y^{(i)}\\log\\left(a^{[2] (i)}\\right) + (1-y^{(i)})\\log\\left(1- a^{[2] (i)}\\right) \\large{)} \\small\\tag{13}$$'''
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        Returns:
        cost -- cross-entropy cost given equation 
        """
        m = Y.shape[1]
        logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
        cost = -1/m * np.sum(logprobs)
        return float(np.squeeze(cost))

    def backward_propagation(self, cache, X, Y):    
        """
            Implement the backward propagation-
            Arguments:
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
            X -- input data of shape (2, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
            Returns:
            grads -- python dictionary containing your gradients with respect to different parameters
        """
        #backpropagation:
        #dZ^2 = A^2 - Y
        #dW^2 = 1/m dZ^2 . A^1^T
        #db^2 = 1/m np.sum(dZ^2,axis=1, keepdims=True)
        #dZ^1 = W^2^T dZ^2 * activation func' (Z^1)  
        #dW^1 = 1/m dZ^1 . X^T
        #db^1 = 1/m np.sum(dZ^1, axis=1, keepdims=True)'''
        m = X.shape[1]
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2)) # with tanh
        dW1 = 1/m * np.dot(dZ1,X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads):

        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        W1 -= self.lr * grads["dW1"]
        b1 -= self.lr * grads["db1"]
        W2 -= self.lr * grads["dW2"]
        b2 -= self.lr * grads["db2"]

        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def fit(self, X, Y):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(0)
        m = X.shape[1]
        input_dim = X.shape[0]
        output_dim = 1 
        costs = []

        self.initialize_parameters()

        for i in range(self.n_iters):
            A2, cache = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(cache, X, Y)
            self.parameters = self.update_parameters(grads)

            # Print the cost every 100 iterations
            if i % 100 == 0:
                costs.append(cost)
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters

    def predict(self, X):
        '''Predict the class labels for the provided data'''
        A2, _ = self.forward_propagation(X)
        return (A2 > 0.5)