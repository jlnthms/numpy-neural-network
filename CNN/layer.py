from typing import Tuple

from CNN.kernel import Kernel
from CNN.activation import *

activation_functions = {
    'relu': relu,
    'tanh': tanh,
    'leaky_relu': leaky_relu,
    # Add more
}


class Layer:
    def __init__(self):
        self.inputs = []
        self.output = []


class ConvLayer(Layer):
    def __init__(self, size: int, kernel_size: Tuple[int, int], activation: str):
        super().__init__()
        self.size = size
        self.kernels = [Kernel(kernel_size) for _ in range(size)]
        self.biases = []

        if activation in activation_functions:
            self.activation = activation_functions[activation]
        else:
            raise ValueError("Invalid activation function name")

    def convolve(self, stride=1, padding=0):
        for image in self.inputs:
            feature_maps = []
            for kernel in self.kernels:
                # each IFM is convolved to a kernel
                feature_maps.append(kernel.convolve(image, stride, padding))
            # the OFM for this kernel is the sum of all convolutions
            output_feature_map = np.sum(feature_maps, axis=0)
            self.output.append(output_feature_map)

    def activate(self):
        for i, feature_map in enumerate(self.output):
            self.output[i] = self.activation(feature_map)


class PoolLayer(Layer):
    def __init__(self, pool_shape: Tuple[int, int], stride=1, method='max'):
        super().__init__()
        self.pool_shape = pool_shape
        self.stride = stride
        self.method = method

    def pool(self):
        if self.method not in ['max', 'average']:
            raise ValueError(f'Non-valid Pooling method: {self.method}')

        for feature_map in self.inputs:
            ph, pw = self.pool_shape
            ih, iw, num_channels = feature_map.shape
            output_h = (ih - ph) // self.stride + 1
            output_w = (iw - pw) // self.stride + 1
            pooled_fm = np.zeros((output_h, output_w, num_channels), dtype=feature_map.dtype)

            for c in range(num_channels):
                for i in range(0, ih - ph + 1, self.stride):
                    for j in range(0, iw - pw + 1, self.stride):
                        region = feature_map[i:i + ph, j:j + pw, c]
                        if self.method == 'max':
                            pooled_fm[i // self.stride, j // self.stride, c] = np.max(region)
                        else:
                            pooled_fm[i // self.stride, j // self.stride, c] = np.mean(region)

            self.output.append(pooled_fm)


class FlatLayer(Layer):
    def __init__(self):
        super().__init__()

    def flatten(self):
        classifier_input = [feature_map.flatten() for feature_map in self.inputs]
        self.output.append(np.concatenate(classifier_input, axis=1))
