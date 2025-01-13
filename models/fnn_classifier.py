import numpy as np

class FNN1Layer:
    def __init__(self, input_dim , output_dim, nunits=2, learning_rate=0.01, n_iters=1000, lambd_reg=0.00, dropout=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nunits = nunits
        self.lr = learning_rate
        self.n_iters = n_iters
        self.parameters = {}
        # regulatization prevents overfitting by penalizing large weights in the cost function
        self.lambd_reg = lambd_reg
        # dropout prevents overffiting because it prevents the network from relying too much on any one unit
        #ensemble-like, with multiple sub-networks being trained
        self.dropout = dropout

    @staticmethod
    def sigmoid(x):
        '''Sigmoid function that works with numpy arrays'''
        return 1 / (1 + np.exp(-x))

    def initialize_parameters_zeros(self):
        '''Initialize weights and bias to zeros
        this results in hidden units to be identical/ symmetric, thus a simple linear model'''
        W1 = np.zeros((self.nunits, self.input_dim))
        b1 = np.zeros((self.nunits, 1))
        W2 = np.zeros((self.output_dim, self.nunits))
        b2 = np.zeros((self.output_dim, 1))
        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def initialize_parameters_random(self):
        '''Initialize weights and bias randomly'''
        W1 = np.random.randn(self.nunits, self.input_dim) * 0.01 
        b1 = np.zeros((self.nunits, 1))
        W2 = np.random.randn(self.output_dim, self.nunits) * 0.01
        b2 = np.zeros((self.output_dim, 1))
        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        
    def initialize_parameters_he(self):
        '''Initialize weights and bias using He initialization
        similar to Xavier initialization, but for ReLU activation functions. '''
        W1 = np.random.randn(self.nunits, self.input_dim) * np.sqrt(2/self.input_dim)
        b1 = np.zeros((self.nunits, 1))
        W2 = np.random.randn(self.output_dim, self.nunits) * np.sqrt(2/self.nunits)
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

        #dropout mask
        if self.dropout:
            D1 = np.random.rand(A1.shape[0], A1.shape[1]) < 1 - self.dropout
            A1 = np.multiply(A1, D1)
            A1 /= 1 - self.dropout #inverted dropout to keep the expected value of the activations the same
    
        Z2 = np.dot(W2, A1) + b2 #(1,number of examples)
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1,
            "D1": D1,
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
        l2_regularization_cost = self.lambd_reg / (2 * m) * np.sum(np.square(self.parameters["W1"]))
        return float(np.squeeze(cost)) + l2_regularization_cost

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

        dA1 = np.dot(W2.T, dZ2)
        if self.dropout:
            D1 = cache["D1"]
            dA1 = np.multiply(dA1, D1)
            dA1 /= 1 - self.dropout

        dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2)) # with tanh
        dW1 = 1/m * np.dot(dZ1,X.T) + self.lambd_reg / m * W1 #derivative with regularization term
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

        self.initialize_parameters_he()

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

    def check_gradient_implementation(X, Y):
        '''Check the implementation of the gradients'''
        np.random.seed(0)
        m = X.shape[1]
        input_dim = X.shape[0]

        nn = NeuralNetwork1Layer(input_dim, 1, nunits=2, learning_rate=0.01, n_iters=1000, lambd=0.00, dropout=None)
        nn.initialize_parameters_he()
        


class FNNClassifier:
    def __init__(self, input_dim, output_dim, layers_dims, activations, learning_rate=0.01, n_iters=1000, lamdb_reg=0.00):
                
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_dims = [input_dim] + layers_dims + [output_dim]
        self.activations = activations
        self.lr = learning_rate
        self.n_iters = n_iters
        self.parameters = {}
        self.lamdb_reg = lamdb_reg

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