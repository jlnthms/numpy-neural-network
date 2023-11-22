from typing import List

from CNN.layer import *
from NeuralNetwork.network import MultiLayerPerceptron
from scipy.signal import convolve2d


class ConvNeuralNet:
    def __init__(self, layers: List[Layer], classifier: MultiLayerPerceptron):
        self.layers = layers
        # TODO: the argument should only be a list of Dense layers (from NeuralNetwork) and the classifier is created
        #  in the constructor, input size is computed from the architecture size reduction and output size from label
        #  size.
        self.classifier = classifier

    def he_initialization(self):
        output_size = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvLayer):
                for kernel in layer.kernels:
                    fan_in = layer.size * output_size
                    std_dev = np.sqrt(2.0 / fan_in)
                    kernel.array = np.random.randn(*kernel.array.shape) * std_dev
                    output_size = layer.size

        self.classifier.he_init()

    def forward(self, image):
        self.layers[0].inputs.append(image)
        for i, layer in enumerate(self.layers):
            if i != 0:
                layer.inputs = self.layers[i - 1].output
            if isinstance(layer, ConvLayer):
                layer.convolve()
                layer.activate()
            elif isinstance(layer, PoolLayer):
                layer.pool()
            elif isinstance(layer, FlatLayer):
                layer.flatten()
            else:
                raise TypeError(f'Invalid or Incomplete layer definition: {layer} of type {type(layer)}')

        self.classifier.forward(self.layers[-1].output[0])

    def backward(self, label):
        # Compute the gradient of the loss with respect to the output of the classifier
        grad_wrt_outputs = self.classifier.backward(label)[2]
        grad_wrt_outputs = np.dot(grad_wrt_outputs, self.classifier.layers[1].get_weights()) * 1

        # Unflatten grad_wrt_outputs to the shape of the last layer's output
        grad_wrt_outputs = grad_wrt_outputs.reshape(np.array(self.layers[-1].inputs).shape)
        grads_k = []

        for layer in self.layers[::-1]:
            if isinstance(layer, ConvLayer):
                # If it's a ConvLayer, we need to compute gradients for kernels and back-propagate the error
                # Create an empty list to store gradients for each kernel in this layer
                kernel_gradients = [np.zeros_like(kernel.array) for kernel in layer.kernels]
                new_gradient_wrt_outputs = np.zeros_like(layer.inputs)
                # Iterate over the feature maps in the current layer
                for i in range(len(layer.output)):
                    # Compute the gradients for this feature map
                    gradient = grad_wrt_outputs[i]
                    # Back-propagate through the activation function
                    gradient = gradient * layer.activation(layer.output[i], derivative=True)
                    # Update kernel gradients using convolution with the corresponding input
                    for j, input_map in enumerate(layer.inputs):
                        kernel_gradients[i] += convolve2d(input_map, gradient, 'valid')
                        # Compute the gradient for the input feature map for the next layer
                        new_gradient_wrt_outputs[j] += convolve2d(gradient, np.rot90(np.rot90(layer.kernels[i].array)), 'full')

                grad_wrt_outputs = new_gradient_wrt_outputs
                grads_k.insert(0, kernel_gradients)

            if isinstance(layer, PoolLayer):
                gradient_maps = np.zeros_like(layer.inputs)

                for i in range(len(layer.inputs)):
                    input_map = layer.inputs[i]
                    grad_wrt_output = grad_wrt_outputs[i]

                    for h in range(0, input_map.shape[0], layer.stride):
                        for w in range(0, input_map.shape[1], layer.stride):
                            if layer.method == 'max':
                                # Find the index of the maximum value in the corresponding pooling region
                                max_index = np.unravel_index(
                                    np.argmax(input_map[h:h + layer.pool_shape[0], w:w + layer.pool_shape[1]]), layer.pool_shape)
                                if len(input_map.shape) > 2:
                                    for c in range(input_map.shape[2]):
                                        gradient_maps[i][h + max_index[0], w + max_index[1], c] = grad_wrt_output[h // layer.stride, w // layer.stride, c]
                                else:
                                    gradient_maps[i][h + max_index[0], w + max_index[1]] = grad_wrt_output[h // layer.stride, w // layer.stride]

                            elif layer.method == 'average':
                                if len(input_map.shape) > 2:
                                    for c in range(input_map.shape[2]):
                                        gradient_maps[i][h:h + layer.pool_shape[0], w:w + layer.pool_shape[1], c] += \
                                            grad_wrt_output[h // layer.stride, w // layer.stride, c] / (layer.pool_shape[0] * layer.pool_shape[1])
                                else:
                                    gradient_maps[i][h:h + layer.pool_shape[0], w:w + layer.pool_shape[1]] += \
                                        grad_wrt_output[h // layer.stride, w // layer.stride] / (layer.pool_shape[0] * layer.pool_shape[1])
                grad_wrt_outputs = gradient_maps
        return grads_k
