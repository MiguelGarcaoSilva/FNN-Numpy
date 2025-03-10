class FNNClassifier:
    def __init__(self, input_dim, output_dim, layers_dims, activations, lamdb_reg=0.00):
                
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_dims = [input_dim] + layers_dims + [output_dim]
        self.activations = activations
        self.parameters = {}
        self.lamdb_reg = lamdb_reg

        assert len(self.layers_dims)  == len(activations) + 2


    @staticmethod
    def softmax(x):
        '''Softmax activation function with cache'''
        exps = np.exp(x - np.max(x)) #subtracting max(x) to avoid numerical instability
        return exps / np.sum(exps, axis=0), x

    @staticmethod
    def relu(x):
        '''Relu activation function with cache'''
        return x * (x > 0), x


    @staticmethod
    def relu_backward(dA, Z):
        '''Relu activation backward'''
        return dA * (Z > 0)


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
        Implement the cost function with frobenius norm regularization
        ||W_l||^2 = sum(W_l^2) = W1^2 + W2^2 + ... + Wn^2 = W^T . W

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10) # to avoid log(0)
        cost = -1/m * np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y))
        # L2 regularization lambda/2m * sum(||W_l||^2)
        l2_regularization_cost = self.lamdb_reg/(2*m) * np.sum([np.sum(np.square(self.parameters['W' + str(l)])) for l in range(1, len(self.layers_dims))])
        return np.squeeze(cost) + l2_regularization_cost


    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T) + (self.lamdb_reg / m) * W #derivative with regularization term
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

    def update_parameters_gd(self, grads, lr) -> None:
        for l in range(1, len(self.layers_dims)):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - lr * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - lr * grads["db" + str(l)]
        return
    
    def update_parameters_momentum(self, grads, lr, v, beta=0.9) -> None:
        for l in range(1, len(self.layers_dims)):
            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)] #smoothes the vertical direction to avoid oscillations
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - lr * v["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - lr * v["db" + str(l)]
        return

    
    def update_parameters_adam(self, grads, lr, v, s, t, beta1=0.9, beta2=0.999, epsilon=10e-8) -> None:
        
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

    def fit(self, X, Y, optimizer, learning_rate, n_iters, seed=0):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(seed)
        costs = []

        self.initialize_parameters()

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

            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            if optimizer == "gd":
                self.update_parameters_gd(grads, learning_rate)
            elif optimizer == "momentum":
                self.update_parameters_momentum(grads, learning_rate, v)
            elif optimizer == "adam":
                t += 1
                self.update_parameters_adam(grads, learning_rate, v, s, t)
            costs.append(cost)

            # Print the cost every 100 iterations
            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters, costs

    def fit_mini_batch(self, X, Y, optimizer, learning_rate, num_epochs, batch_size, seed=0):
        np.random.seed(seed)
        costs = []

        #shuffle the data
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        X = X[:, permutation]
        Y = Y[:, permutation]

        self.initialize_parameters()

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
            seed += 1
            np.random.seed(seed)
            permutation = list(np.random.permutation(m))
            X = X[:, permutation]
            Y = Y[:, permutation]

            for j in range(0, X.shape[1], batch_size):
                X_batch = X[:, j:j+batch_size]
                Y_batch = Y[:, j:j+batch_size]

                AL, caches = self.forward_propagation(X_batch)                
                cost = self.compute_cost(AL, Y_batch)
                grads = self.backward_propagation(AL, Y_batch, caches)
                if optimizer == "gd":
                    self.update_parameters_gd(grads, learning_rate)
                elif optimizer == "momentum":
                    self.update_parameters_momentum(grads, learning_rate, v)
                elif optimizer == "adam":
                    t += 1
                    self.update_parameters_adam(grads, learning_rate, v, s, t)

            costs.append(cost)
            # Print the cost every 100 epoch
            if i % 100 == 0:
                print(f'Cost after epoch {i}: {cost}')

        return self.parameters, costs



    def predict(self, X):
        '''Predict the class labels for the provided data'''
        AL, _ = self.forward_propagation(X)
        return (AL > 0.5)