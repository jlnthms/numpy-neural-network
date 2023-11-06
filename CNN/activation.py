import numpy as np


def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


def leaky_relu(x, alpha=0.01, derivative=False):
    if derivative:
        return np.where(x > 0, 1, alpha)
    return np.maximum(alpha * x, x)
