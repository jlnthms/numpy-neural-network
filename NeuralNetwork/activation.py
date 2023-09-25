import numpy as np


def identity(x):
    return x


def relu(x, derivative=False):
    if derivative:
        return 0 if x < 0 else 1
    return max(0, x)


# TODO: not sure its correct (vector argument like softmax)
# def sigmoid(x, derivative=False):
#     sig = 1 / (1 + np.exp(-x))
#     if derivative:
#         return sig * (1-sig)
#     return sig


def softmax(x, derivative=False):
    e_x = np.exp(x - np.max(x))
    softmax_v = e_x / e_x.sum()
    if derivative:
        num_classes = len(softmax_v)
        gradient = np.eye(num_classes)
        gradient = np.where(gradient, softmax_v * (1 - softmax_v), -softmax_v[:, np.newaxis] * softmax_v)

        return gradient
    return softmax_v



