import numpy as np

def single_layer_nn(X, w, b):
    # Ensure X is a 2D array
    X = np.atleast_2d(X)
    
    # Compute the linear combination
    z = np.dot(X, w) + b  # Shape: (n_samples,)
    
    # Apply the sigmoid activation function
    y = 1 / (1 + np.exp(-z))
    
    return y
