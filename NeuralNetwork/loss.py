from NeuralNetwork.activation import *


def cross_entropy(y_pred: list, y_true: list, derivative=False):
    y_pred = softmax(y_pred)
    y_true = np.array(y_true)
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return - y_true / y_pred

    return -np.sum(y_true * np.log(y_pred))
