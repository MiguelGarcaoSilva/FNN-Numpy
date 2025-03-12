
import numpy as np

class FNNClassifier:
    def __init__(self, input_dim, output_dim, layers_dims, activations, lamdb_reg=0.00):
        '''
        Initialize the parameters of the feedforward neural network

        Arguments:
        input_dim -- size of the input layer
        output_dim --  size of the ouput layer
        layers_dims -- list containing the size of each hidden layer
        activations -- list containing the activation functions for each hidden layer
        lamdb_reg -- regularization parameter
        '''
     
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_dims = [input_dim] + layers_dims + [self.output_dim]

        if output_dim == 1:
            self.activations = activations + ["sigmoid"]
        else:
            self.activations = activations + ["softmax"]

        self.lamdb_reg = lamdb_reg  # avoid overfitting by penalizing large weights

        assert len(self.layers_dims)  == len(self.activations) + 1
        assert self.output_dim >= 1

        for activation in activations:
            assert activation in ["relu", "tanh"]

        self.parameters = {}
        for l in range(1, len(self.layers_dims)):
            # Weights are indexed from 1 to L-1, where L is the number of layers
            # e.g., layers_dims = [3, 5, 1], W1 -- of shape (5, 3), W2 -- of shape (1, 5)
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    @staticmethod
    def sigmoid(x):
        '''Sigmoid activation function'''
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        '''Softmax activation function with cache'''
        exps = np.exp(x - np.max(x)) #subtracting max(x) to avoid numerical instability
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def relu(x):
        '''Relu activation function'''
        return x * (x > 0)

    @staticmethod
    def tanh(x):
        '''Tanh activation function'''
        return np.tanh(x)

    @staticmethod
    def relu_backward(dA, Z):
        '''Relu activation backward'''
        return dA * (Z > 0)

    @staticmethod
    def tanh_backward(dA, Z):
        '''Tanh activation backward'''
        return dA * (1 - np.power(np.tanh(Z), 2))


    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or X is first layer): (prev_layer_size, m)
        W -- weights matrix: numpy array of shape (layer_size, prev_layer_size)
        b -- bias vector, numpy array of shape (layer_size, 1)

        Returns:
        Z -- the input of the activation function (pre-activation parameter) 
        cache -- "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        Z = np.dot(W, A) + b 
        return Z, (A, W, b)

    def linear_activation_forward(self, A_prev, W, b, activation_func):
        """
        Arguments:
        A_prev -- activations from previous layer (or input data): (prev_layer_size, m)
        W -- weights matrix: numpy array of shape (layer_size, prev_layer_size)
        b -- bias vector, numpy array of shape (layer_size, 1)
        activation_func -- the activation function from current layer
        
        Returns:
        A-- the output of the activation function
        cache -- a tuple containing "linear_cache" and "activation_cache"
        """
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A = activation_func(Z)  
        return A, (linear_cache, Z)

    
    def forward(self, X):
        """
        Implement forward propagation for the [LINEAR->ActivationFunc]*(L-1)->LINEAR->SIGMOID/Softmax
        
        Arguments:
        X -- data, numpy array of shape (input size, m)
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (L indexed from 0 to L-1)
        """ 
        caches = []
        A = X
        L = len(self.layers_dims)

        for l in range(1, L-1): # all hidden layers except the output layer
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            if self.activations[l-1] == "relu":
                A, cache = self.linear_activation_forward(A_prev, W, b, self.relu)
                caches.append(cache)
            else: #tanh
                A, cache = self.linear_activation_forward(A_prev, W, b, self.tanh)
                caches.append(cache) 

        if self.activations[-1] == "sigmoid":
            AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L-1)], self.parameters['b' + str(L-1)], self.sigmoid)
        else:
            AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L-1)], self.parameters['b' + str(L-1)], self.softmax)

        caches.append(cache)
        return AL, caches

    @staticmethod
    def binary_crossentropy(AL, Y, m):
        """Compute binary cross-entropy cost.
        
        Parameters:
            AL (np.ndarray): Output layer activation.
            Y (np.ndarray): True labels of shape (1, m).
            m (int): Number of examples.
        
        Returns:
            cost -- cross-entropy cost
         """
        return -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    
    @staticmethod
    def categorical_crossentropy(AL, Y, m):
        """
        Implement the cost function with cross-entropy loss
        Arguments:
        AL -- probability vector corresponding to label predictions, shape (output_dim, m)
        Y -- true "label" vector  shape (output_dim, m)
        m -- number of examples

        Returns:
        cost -- cross-entropy cost
        """
        return -np.sum(Y * np.log(AL + 1e-9)) / m
    
    def compute_cost(self, AL, Y):
        """
        Computes the cost function with regularization

        Arguments:
        AL -- probability vector corresponding to label predictions, shape (output_dim, m)
        Y -- true "label" vector shape (output_dim, m)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10) # to avoid log(0)
        cost = self.binary_crossentropy(AL, Y, m) if self.output_dim == 1 else self.categorical_crossentropy(AL, Y, m)
        # L2 regularization lambda/2m * sum(||W_l||^2)
        l2_regularization_cost = self.lamdb_reg/(2*m) * np.sum([np.sum(np.square(self.parameters['W' + str(l)])) for l in range(1, len(self.layers_dims))])
        return np.squeeze(cost) + l2_regularization_cost
        

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l).
        Given dZ, dW is calculated as dZ . A_prev.T, db as 1/m * sum(dZ) and dA_prev as W.T . dZ

        Arguments:
        dZ -- Gradient of the cost w.r.t linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost w.r.t activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost w.r.t W (current layer l), same shape as W
        db -- Gradient of the cost w.r.t b (current layer l), same shape as b
        """
    
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T) + (self.lamdb_reg / m) * W #derivative with regularization term
        db = 1/m * np.sum(dZ, axis=1, keepdims=True) 
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        Combines the linear_backward and the backward step for the activation function

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache)
        activation -- the activation to be used in this layer

        Returns:
        dA_prev -- Gradient of the cost w.r.t activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost w.r.t W (current layer l), same shape as W
        db -- Gradient of the cost w.r.t b (current layer l), same shape as b
        """

        linear_cache, activation_cache = cache # cache from forward pass

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        else: #tanh
            dZ = self.tanh_backward(dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backward(self, AL, Y, caches):    
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID/Softmax
        
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

        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        linear_cache, _ = caches[-1]  # the cache of the output layer

        dZ = AL - Y # Simplified derivative of cross-entropy loss with sigmoid and softmax

        dA_prev_temp, dW_temp, db_temp= self.linear_backward(dZ, linear_cache)
            
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp 
        grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (activation -> linear) gradients.
            current_cache = caches[l] # indexed from 0 to L-1
            activation = self.activations[l] # indexed from 0 to L-1
            current_grads = grads["dA" + str(l+1)] # indexed from 1 to L

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(current_grads, current_cache, activation)
            grads["dA" + str(l)] = dA_prev_temp 
            grads["dW" + str(l + 1)] = dW_temp # indexed from 1 to L
            grads["db" + str(l + 1)] = db_temp # indexed from 1 to L

        return grads

    def update_parameters_gd(self, grads, lr) -> None:
        for l in range(1, len(self.layers_dims)):
            self.parameters["W" + str(l)] -= lr * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= lr * grads["db" + str(l)]
        return
    
    def update_parameters_momentum(self, grads, lr, v, beta=0.9) -> None:
        for l in range(1, len(self.layers_dims)):
            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)] #smoothes the vertical direction to avoid oscillations
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - lr * v["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - lr * v["db" + str(l)]
        return

    
    def update_parameters_adam(self, grads, lr, v, s, t, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        
        v_corrected = {}
        s_corrected = {}
        for l in range(1, len(self.layers_dims)):
            
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)] # momentum
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)]) # RMSprop
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t) # bias correction
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - lr * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - lr * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
        return

    def fit(self, X, Y, optimizer, learning_rate, n_iters, verbose=True):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(0)
        costs = []

        if optimizer == "momentum":
            v = {}
            for l in range(1, len(self.layers_dims)):
                v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
        elif optimizer == "adam":
            v = {}
            s = {}
            t = 0
            for l in range(1, len(self.layers_dims)):
                v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
                s["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                s["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])

        for i in range(n_iters):

            AL, caches = self.forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward(AL, Y, caches)
            if optimizer == "gd":
                self.update_parameters_gd(grads, learning_rate)
            elif optimizer == "momentum":
                self.update_parameters_momentum(grads, learning_rate, v)
            elif optimizer == "adam":
                t += 1
                self.update_parameters_adam(grads, learning_rate, v, s, t)
            costs.append(cost)

            # Print the cost every 100 iterations
            if verbose and i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return costs

    def fit_mini_batch(self, X, Y, optimizer, learning_rate, num_epochs, batch_size, verbose=True):
        np.random.seed(0)
        costs = []

        #shuffle the data
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        X = X[:, permutation]
        Y = Y[:, permutation]

        if optimizer == "momentum":
            v = {}
            for l in range(1, len(self.layers_dims)):
                v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
        elif optimizer == "adam":
            v = {}
            s = {}
            t = 0
            for l in range(1, len(self.layers_dims)):
                v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
                s["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
                s["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])

        for i in range(num_epochs):

            X = X[:, permutation]
            Y = Y[:, permutation]

            for j in range(0, X.shape[1], batch_size):
                X_batch = X[:, j:j+batch_size]
                Y_batch = Y[:, j:j+batch_size]

                AL, caches = self.forward(X_batch)                
                cost = self.compute_cost(AL, Y_batch)
                grads = self.backward(AL, Y_batch, caches)
                if optimizer == "gd":
                    self.update_parameters_gd(grads, learning_rate)
                elif optimizer == "momentum":
                    self.update_parameters_momentum(grads, learning_rate, v)
                elif optimizer == "adam":
                    t += 1
                    self.update_parameters_adam(grads, learning_rate, v, s, t)

            costs.append(cost)
            # Print the cost every 100 epoch
            if verbose and i % 100 == 0:
                print(f'Cost after epoch {i}: {cost}')

        return costs


    def predict(self, X):
        AL, _ = self.forward(X)
        return np.argmax(AL, axis=0) if self.output_dim > 1 else (AL > 0.5)

