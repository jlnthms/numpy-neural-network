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

        self.classifier.he_init()

    def forward(self, image):
        if self.layers[0].inputs:
            self.layers[0].inputs = []
        self.layers[0].inputs.append(image)
        for i, layer in enumerate(self.layers):
            # clear output from previous pass
            if layer.output:
                layer.output = []
            # replace inputs with output of the previous layer (if previous)
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
        grad_class_w, grad_class_b, grad_wrt_outputs = self.classifier.backward(label)
        grad_wrt_outputs = np.dot(grad_wrt_outputs, self.classifier.layers[1].get_weights()) * 1

        # Unflatten grad_wrt_outputs to the shape of the last layer's output
        grad_wrt_outputs = grad_wrt_outputs.reshape(np.array(self.layers[-1].inputs).shape)
        grads_k = []

        for layer in self.layers[::-1]:
            if isinstance(layer, ConvLayer):
                grads_k, grad_wrt_outputs = layer.backward(grads_k, grad_wrt_outputs)

            if isinstance(layer, PoolLayer):
                grad_wrt_outputs = layer.backward(grad_wrt_outputs)
        return grad_class_w, grad_class_b, grads_k
