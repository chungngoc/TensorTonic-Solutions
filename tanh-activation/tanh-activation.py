import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    y = - np.array(x)
    
    return (np.exp(x) - np.exp(y)) / (np.exp(x) + np.exp(y))