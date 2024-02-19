"""
Layer

This module contains the base Layer class and derived classes
ConvLayer, PoolLayer and FlatLayer.

Author: Julien THOMAS
Date: November 04, 2023
"""

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
    """
    A Layer is a step in the network that performs a specific operation. It could be either a kernel convolution,
    a pooling operation or flattening. Each layer takes the output (modified images) of the previous layer as input.

    Attributes:
        inputs (List[np.ndarray]): array of incoming images from the previous layer.
        output (List[np.ndarray]): transformed input images.

    Derived Classes:
        ConvLayer: Convolutional layers
        PoolLayer: Pooling layers
        FlatLayer: Flattening layers
    """
    def __init__(self):
        self.inputs = []
        self.output = []


class ConvLayer(Layer):
    """
    Convolutional layers apply kernel convolution to the input images. It is composed of a set of kernels filtering the
    input to extract features. The class inherits the inputs and output attribute from base Layer class (see above).

    Attributes:
        size (int): Number of kernels in the layer.
        kernels (List[Kernel]): List of kernels of chosen dimensions.
        biases (List): List of biases, each one applies to the corresponding kernel.

    Methods:
        convolve(stride, padding): kernel convolution.
        activate(): applying layer activation function to pixels on feature maps.
        backward(grads_k, grad_wrt_outputs): backward pass of gradients w.r.t kernels of the layer.
    """
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
        """
        Each kernel is convolved with every input feature map. The results from these convolution are summed into the
        output feature map corresponding to each kernel. There are as many output feature maps as there are kernels in
        the layer.

        Args:
            stride(int): number of pixels the kernel slides over at each iteration of the convolution.
            padding(int): number of added pixels on both dimensions (usually when kernel size does
            not divide input size).

        Returns:
            None
        """
        for kernel in self.kernels:
            feature_maps = [kernel.convolve(image, stride, padding) for image in self.inputs]
            output_feature_map = np.sum(feature_maps, axis=0)
            self.output.append(output_feature_map)

    def activate(self):
        """
        Applies layer activation function to the images to introduce non-linearity in the transformation and extract
        more complex features.

        Returns:
            None
        """
        for i, feature_map in enumerate(self.output):
            # TODO: add the use of biases
            self.output[i] = self.activation(feature_map)

    def backward(self, grads_k, grad_wrt_outputs):
        """
        Computation of the gradients w.r.t outputs and kernels to later update the kernels' pixel values in
        the direction of the decreasing gradients.

        Args:
            grads_k (List[np.ndarray]): list of gradient matrices w.r.t kernels.
            grad_wrt_outputs (List[np.ndarray]): list of gradient matrices w.r.t output feature maps.

        Returns:
            grads_k, grad_wrt_outputs (Tuple): new gradients for the next layer's backward pass.
        """
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
    """
    Pooling layers are responsible for dimensionality reduction to focus on meaningful details
    of the input and reduce the computational workload of the training by down-sampling feature maps.
    The two mainly used methods for this are average pooling and max pooling, both these methods are
    implemented in the class.

    Attributes:
        pool_shape (Tuple[int, int]): height, width of the pooling window.
        stride (int): number of pixels the pooling window slides over during the operation.
        method (str): 'max' or 'average', defines the chosen pooling method for the layer.

    Methods:
        pool(): performs pooling with the chosen method.
        backward(): performs the backward pass according to the chosen methods.
    """
    def __init__(self, pool_shape: Tuple[int, int], stride=1, method='max'):
        super().__init__()
        self.pool_shape = pool_shape
        self.stride = stride
        self.method = method

    def pool(self):
        """
        This method applies either max or average pooling to the input of the layer. For each image,
        the pooling window strides over to output either the maximum or the average value of all its
        composing pixels.

        Returns:
            None
        """
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
        """
        Backward pass for the pooling layer. In the case of average pooling, the gradient is distributed equally
        across the pooling window, and in the case of max pooling only the index of the original max is granted
        the gradient, as it is the only responsible for the loss.

        Args:
            grad_wrt_outputs (List[np.ndarray]): list of gradient matrices w.r.t output feature maps.
        """
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
    """
    The flattening layer is usually located just before the dense layers to flatten the feature
    maps into a single dimension array to fit the architecture of the following ANN.

    Methods:
        flatten(): flattens the input of the layer.
    """
    def __init__(self):
        super().__init__()

    def flatten(self):
        """
        Each input feature map is flattened by row into a single column, which are then concatenated
        into the input array for the dense layers.

        Returns:
            None
        """
        classifier_input = [feature_map.flatten() for feature_map in self.inputs]
        self.output.append(np.concatenate(classifier_input))
