from typing import Tuple
import numpy as np

from CNN.kernel import Kernel


class Layer:
    def __init__(self):
        self.inputs = []
        self.output = []


class ConvLayer(Layer):
    def __init__(self, size: int, kernel_size: Tuple[int, int], activation: str):
        super().__init__()
        self.size = size
        self.kernels = [Kernel(kernel_size) for _ in range(size)]

        self.activation = activation
        if activation == 'leaky_relu':
            pass
        else:
            pass

    def convolve(self, stride=1, padding=0):
        for image in self.inputs:
            for kernel in self.kernels:
                # TODO: add activation (e.g Leaky ReLu etc)
                output_feature_map = kernel.convolve(image, stride, padding)
                self.output.append(output_feature_map)


class PoolLayer(Layer):
    def __init__(self, pool_shape: Tuple[int, int], stride=1):
        super().__init__()
        self.pool_shape = pool_shape
        self.stride = stride

    def pool(self, method: str):
        if method not in ['max', 'average']:
            raise ValueError(f'Non-valid Pooling method: {method}')

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
                        if method == 'max':
                            pooled_fm[i // self.stride, j // self.stride, c] = np.max(region)
                        else:
                            pooled_fm[i // self.stride, j // self.stride, c] = np.mean(region)

            self.output.append(pooled_fm)


class FlatLayer(Layer):
    def __init__(self):
        super().__init__()

    def flatten(self):
        for feature_map in self.inputs:
            self.output.append(feature_map.flatten())
