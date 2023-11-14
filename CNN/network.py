from typing import List

from CNN.layer import *
from NeuralNetwork.network import MultiLayerPerceptron


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

    def forward(self, image):
        self.layers[0].inputs.append(image)
        for i, layer in enumerate(self.layers):
            if i != 0:
                layer.inputs = self.layers[i-1].output
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

        # Unflatten grad_wrt_outputs to the shape of the last layer's output
        grad_wrt_outputs = grad_wrt_outputs.reshape(np.array(self.layers[-1].inputs).shape)
        grads_k = []

        for layer in self.layers[::-1]:
            if isinstance(layer, ConvLayer):
                # If it's a ConvLayer, we need to compute gradients for kernels and back-propagate the error
                # Create an empty list to store gradients for each kernel in this layer
                kernel_gradients = [np.zeros_like(kernel.array) for kernel in layer.kernels]
                # Iterate over the feature maps in the current layer
                for i in range(len(layer.output)):
                    # Compute the gradients for this feature map
                    gradient = grad_wrt_outputs[i]
                    # Back-propagate through the activation function
                    gradient = gradient * layer.activation(layer.output[i], derivative=True)
                    # Update kernel gradients using convolution with the corresponding input
                    for j in range(len(layer.kernels)):
                        input_map = layer.inputs[i]
                        kernel_gradients[j] += np.rot90(np.rot90(np.convolve(input_map, gradient, 'valid')))
                    # Compute the gradient for the input feature map for the next layer
                    grad_wrt_outputs[i] = np.zeros_like(grad_wrt_outputs[i])
                    for j in range(len(layer.kernels)):
                        grad_wrt_outputs[i] += np.convolve(layer.kernels[j].array, gradient, 'full')

                grads_k.insert(0, kernel_gradients)

            if isinstance(layer, PoolLayer):
                if layer.method == 'max':
                    gradient_map = np.zeros_like(layer.inputs[0])
                    for i in range(len(layer.output)):
                        for j in range(len(layer.output[i])):
                            for k in range(len(layer.output[i][j])):
                                region = layer.inputs[i][j * layer.stride:(j * layer.stride) + layer.pool_shape[0],
                                         k * layer.stride:(k * layer.stride) + layer.pool_shape[1]]
                                max_val = np.max(region)
                                gradient_map[j * layer.stride:(j * layer.stride) + layer.pool_shape[0],
                                k * layer.stride:(k * layer.stride) + layer.pool_shape[1]] = (layer.output[i][j][k] == max_val) * grad_wrt_outputs[i][j][k]
                    grad_wrt_outputs = gradient_map

                elif layer.method == 'average':
                    gradient_map = np.zeros_like(layer.inputs[0])
                    for i in range(len(layer.output)):
                        for j in range(len(layer.output[i])):
                            for k in range(len(layer.output[i][j])):
                                gradient_map[j * layer.stride:(j * layer.stride) + layer.pool_shape[0],
                                k * layer.stride:(k * layer.stride) + layer.pool_shape[1]] += grad_wrt_outputs[i][j][k] / (layer.pool_shape[0] * layer.pool_shape[1])
                    grad_wrt_outputs = gradient_map

        return grads_k
