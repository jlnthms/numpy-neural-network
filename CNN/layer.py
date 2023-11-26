from typing import Tuple

from CNN.kernel import Kernel
from CNN.activation import *

from scipy.signal import convolve2d

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
        for kernel in self.kernels:
            feature_maps = [kernel.convolve(image, stride, padding) for image in self.inputs]
            output_feature_map = np.sum(feature_maps, axis=0)
            self.output.append(output_feature_map)

    def activate(self):
        for i, feature_map in enumerate(self.output):
            self.output[i] = self.activation(feature_map)

    def backward(self, grads_k, grad_wrt_outputs):
        # Create an empty list to store gradients for each kernel in this layer
        kernel_gradients = [np.zeros_like(kernel.array) for kernel in self.kernels]
        new_gradient_wrt_outputs = np.zeros_like(self.inputs)
        # Iterate over the feature maps in the current layer
        for i in range(len(self.output)):
            # Compute the gradients for this feature map
            gradient = grad_wrt_outputs[i]
            # Back-propagate through the activation function
            gradient = gradient * self.activation(self.output[i], derivative=True)
            # Update kernel gradients using convolution with the corresponding input
            for j, input_map in enumerate(self.inputs):
                kernel_gradients[i] += convolve2d(input_map, gradient, 'valid')
                # Compute the gradient for the input feature map for the next layer
                new_gradient_wrt_outputs[j] += convolve2d(gradient, np.rot90(np.rot90(self.kernels[i].array)), 'full')

        grad_wrt_outputs = new_gradient_wrt_outputs
        grads_k.insert(0, kernel_gradients)
        return grads_k, grad_wrt_outputs


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

    def backward(self, grad_wrt_outputs):
        gradient_maps = np.zeros_like(self.inputs)

        for i in range(len(self.inputs)):
            input_map = self.inputs[i]
            grad_wrt_output = grad_wrt_outputs[i]

            for h in range(0, input_map.shape[0], self.stride):
                for w in range(0, input_map.shape[1], self.stride):
                    if self.method == 'max':
                        # Find the index of the maximum value in the corresponding pooling region
                        max_index = np.unravel_index(
                            np.argmax(input_map[h:h + self.pool_shape[0], w:w + self.pool_shape[1]]), self.pool_shape)
                        if len(input_map.shape) > 2:
                            for c in range(input_map.shape[2]):
                                gradient_maps[i][h + max_index[0], w + max_index[1], c] = grad_wrt_output[
                                    h // self.stride, w // self.stride, c]
                        else:
                            gradient_maps[i][h + max_index[0], w + max_index[1]] = grad_wrt_output[
                                h // self.stride, w // self.stride]

                    elif self.method == 'average':
                        if len(input_map.shape) > 2:
                            for c in range(input_map.shape[2]):
                                gradient_maps[i][h:h + self.pool_shape[0], w:w + self.pool_shape[1], c] += \
                                    grad_wrt_output[h // self.stride, w // self.stride, c] / (
                                            self.pool_shape[0] * self.pool_shape[1])
                        else:
                            gradient_maps[i][h:h + self.pool_shape[0], w:w + self.pool_shape[1]] += \
                                grad_wrt_output[h // self.stride, w // self.stride] / (
                                        self.pool_shape[0] * self.pool_shape[1])
        grad_wrt_outputs = gradient_maps
        return grad_wrt_outputs


class FlatLayer(Layer):
    def __init__(self):
        super().__init__()

    def flatten(self):
        classifier_input = [feature_map.flatten() for feature_map in self.inputs]
        self.output.append(np.concatenate(classifier_input))
