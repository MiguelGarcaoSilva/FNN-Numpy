import numpy as np

#check gradient for a multi-layer neural network with relu activation
def compute_cost_external(parameters, X, Y, layers_dims, lambd_reg):
    """
    Compute cost externally for a multi-layer neural network.
    
    Arguments:
    parameters -- dictionary containing weights and biases for each layer.
    X -- input data, shape (input size, number of examples)
    Y -- true "label" vector, shape (output size, number of examples)
    layers_dims -- list of layer dimensions
    lambd_reg -- regularization parameter
    
    Returns:
    cost -- computed cost
    """
    m = Y.shape[1]
    A = X

    # Forward pass
    L = len(layers_dims) - 1
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A) + b
        A = np.maximum(0, Z)  # ReLU activation

    # Output layer (Sigmoid activation)
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z = np.dot(W, A) + b
    AL = 1 / (1 + np.exp(-Z))  # Sigmoid activation

    # Compute cost with regularization
    AL = np.clip(AL, 1e-10, 1 - 1e-10)  # Avoid log(0)
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    l2_reg_cost = lambd_reg / (2 * m) * np.sum([np.sum(np.square(parameters['W' + str(l)])) for l in range(1, L+1)])
    return np.squeeze(cost) + l2_reg_cost


def flatten_parameters_and_gradients(parameters, gradients):
    """
    Flatten parameters and gradients into vectors.
    
    Arguments:
    parameters -- dictionary of weights and biases
    gradients -- dictionary of gradients
    
    Returns:
    theta_flat -- flattened parameter vector
    grad_flat -- flattened gradient vector
    shapes -- list of shapes for reconstructing original parameters
    """
    theta_flat, grad_flat, shapes = [], [], []
    for key in parameters.keys():
        theta_flat.extend(parameters[key].flatten())
        grad_flat.extend(gradients["d"+key].flatten())
        shapes.append(parameters[key].shape)
    return np.array(theta_flat), np.array(grad_flat), shapes

def reconstruct_parameters(theta_flat, shapes):
    """
    Reconstruct parameters from flattened vector.
    
    Arguments:
    theta_flat -- flattened parameter vector
    shapes -- list of parameter shapes
    
    Returns:
    parameters -- dictionary of reconstructed parameters
    """
    parameters = {}
    idx = 0
    for i, shape in enumerate(shapes):
        size = np.prod(shape)
        key = 'W' + str(i // 2 + 1) if i % 2 == 0 else 'b' + str(i // 2 + 1)
        parameters[key] = theta_flat[idx:idx + size].reshape(shape)
        idx += size
    return parameters

def gradient_check(model, X, Y, epsilon=1e-7):
    """
    Perform gradient checking for a multi-layer neural network.

    Arguments:
    model -- instance of LogisticRegression or FFNNClassifier
    X -- input data, shape (input size, number of examples)
    Y -- true "label" vector, shape (output size, number of examples)
    epsilon -- small perturbation to compute numerical gradient

    Returns:
    difference -- difference between analytical and numerical gradients
    """
    # Forward pass to get initial gradients
    AL, caches = model.forward_propagation(X)
    gradients = model.backward_propagation(AL, Y, caches)
    parameters = model.parameters
    layers_dims = model.layers_dims
    lambd_reg = model.lamdb_reg

    # Flatten parameters and gradients
    theta_flat, grad_flat, shapes = flatten_parameters_and_gradients(parameters, gradients)
    num_parameters = len(theta_flat)
    gradapprox = np.zeros_like(theta_flat)

    # Compute gradapprox
    for i in range(num_parameters):
        theta_plus = np.copy(theta_flat)
        theta_minus = np.copy(theta_flat)
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        # Reconstruct parameters
        params_plus = reconstruct_parameters(theta_plus, shapes)
        params_minus = reconstruct_parameters(theta_minus, shapes)

        # Compute costs
        J_plus = compute_cost_external(params_plus, X, Y, layers_dims, lambd_reg)
        J_minus = compute_cost_external(params_minus, X, Y, layers_dims, lambd_reg)

        gradapprox[i] = (J_plus - J_minus) / (2 * epsilon)

    # Compute difference
    numerator = np.linalg.norm(grad_flat - gradapprox)
    denominator = np.linalg.norm(grad_flat) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print(f"Backward propagation might have issues. Difference = {difference}")
    else:
        print(f"Backward propagation works well! Difference = {difference}")
    return difference
