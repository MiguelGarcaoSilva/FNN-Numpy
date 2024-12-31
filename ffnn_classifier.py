import numpy as np

class FFNNClassifier:
    def __init__(self, input_dim, output_dim, layers_dims, activations, learning_rate=0.01, n_iters=1000):
                
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_dims = [input_dim] + layers_dims + [output_dim]
        self.activations = activations
        self.lr = learning_rate
        self.n_iters = n_iters
        self.parameters = {}

        assert len(self.layers_dims)  == len(activations) + 2

    @staticmethod
    def sigmoid(x):
        '''Sigmoid activation function with cache'''
        return 1 / (1 + np.exp(-x)), x


    @staticmethod
    def relu(x):
        '''Relu activation function with cache'''
        return x * (x > 0), x

    @staticmethod
    def tanh(x):
        '''Tanh activation function with cache'''
        return np.tanh(x), x

    @staticmethod
    def sigmoid_backward(dA, Z):
        '''Sigmoid activation backward'''
        s = 1 / (1 + np.exp(-Z))
        return dA * s * (1 - s)

    @staticmethod
    def relu_backward(dA, Z):
        '''Relu activation backward'''
        return dA * (Z > 0)

    @staticmethod
    def tanh_backward(dA, Z):
        '''Tanh activation backward'''
        return dA * (1 - np.power(np.tanh(Z), 2))


    def initialize_parameters(self):
        '''Initialize weights and bias randomly
        initializing to zero results in hidden units to be identical'''

        parameters = {}

        
        for l in range(1, len(self.layers_dims)):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
            
        self.parameters = parameters


    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or X is first layer): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation_func):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = activation_func(Z)  
        return A, (linear_cache, activation_cache)

    
    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->ActivationFunc]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (L indexed from 0 to L-1)
        """ 
        caches = []
        A = X
        L = len(self.layers_dims) 

        for l in range(1, L-1):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            if self.activations[l-1] == "relu":
                A, cache = self.linear_activation_forward(A_prev, W, b, self.relu)
                caches.append(cache)
            elif self.activations[l-1] == "tanh":
                A, cache = self.linear_activation_forward(A_prev, W, b, self.tanh)
                caches.append(cache)
            else: #sigmoid
                A, cache = self.linear_activation_forward(A_prev, W, b, self.sigmoid)
                caches.append(cache)     

        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L-1)], self.parameters['b' + str(L-1)], self.sigmoid)
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10) # to avoid log(0)
        cost = -1/m * np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y))
        return np.squeeze(cost)


    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = self.tanh_backward(dA, activation_cache)
        else: # sigmoid
            dZ = self.sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):    
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- label vector 
        caches -- list of caches containing:
                  every cache of each layer (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                  the cache of output layer caches[L-1]
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        #backpropagation:
        #input: dA
        #dZ = daA * g'(Z), by the chain rule
        #dW = 1/m dZ . A.T
        #db = 1/m np.sum(dZ, axis=1, keepdims=True)
        #dA = W.T . dZ
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # solve invalid value encountered in true_divide
        AL = np.clip(AL, 1e-10, 1 - 1e-10) # to avoid log(0)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL for sigmoid
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, self.sigmoid)
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (activation -> linear) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, self.activations[l-1])
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads) -> None:
        for l in range(1, len(self.layers_dims)):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.lr * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.lr * grads["db" + str(l)]
        return

    def fit(self, X, Y, seed=None):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(seed)
        costs = []

        self.initialize_parameters()

        for i in range(self.n_iters):

            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads)

            # Print the cost every 100 iterations
            if i % 100 == 0:
                costs.append(cost)
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters

    def predict(self, X):
        '''Predict the class labels for the provided data'''
        AL, _ = self.forward_propagation(X)
        return (AL > 0.5)