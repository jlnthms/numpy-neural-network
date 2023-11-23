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
        # clear output from previous pass
        if self.output:
            self.output = []
        for kernel in self.kernels:
            feature_maps = []
            for image in self.inputs:
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
        # clear output from previous pass
        if self.output:
            self.output = []
        if self.method not in ['max', 'average']:
            raise ValueError(f'Non-valid Pooling method: {self.method}')

        for feature_map in self.inputs:
            ph, pw = self.pool_shape
            ih, iw = feature_map.shape[0], feature_map.shape[1]
            output_h = (ih - ph) // self.stride + 1
            output_w = (iw - pw) // self.stride + 1
            if len(feature_map.shape) > 2:
                pooled_fm = np.zeros((output_h, output_w, len(feature_map.shape)), dtype=feature_map.dtype)
                num_channels = feature_map.shape[2]
            else:
                pooled_fm = np.zeros((output_h, output_w), dtype=feature_map.dtype)
                num_channels = 1
            for c in range(num_channels):
                for i in range(0, ih - ph + 1, self.stride):
                    for j in range(0, iw - pw + 1, self.stride):
                        region = feature_map[i:i + ph, j:j + pw, c] if num_channels > 1 \
                            else feature_map[i:i + ph, j:j + pw]
                        if self.method == 'max':
                            if num_channels > 1:
                                pooled_fm[i // self.stride, j // self.stride, c] = np.max(region)
                            else:
                                pooled_fm[i // self.stride, j // self.stride] = np.max(region)
                        else:
                            if num_channels > 1:
                                pooled_fm[i // self.stride, j // self.stride, c] = np.mean(region)
                            else:
                                pooled_fm[i // self.stride, j // self.stride] = np.mean(region)

            self.output.append(pooled_fm)


class FlatLayer(Layer):
    def __init__(self):
        super().__init__()

    def flatten(self):
        classifier_input = [feature_map.flatten() for feature_map in self.inputs]
        if self.output:
            self.output.pop()
        self.output.append(np.concatenate(classifier_input))
