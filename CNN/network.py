from typing import List

from CNN.layer import *
from NeuralNetwork.network import MultiLayerPerceptron


class ConvNeuralNet:
    def __init__(self, layers: List[Layer], classifier: MultiLayerPerceptron):
        self.layers = layers
        self.classifier = classifier

    def he_initialization(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvLayer):
                for kernel in layer.kernels:
                    fan_in = len(self.layers[i-1].output)
                    std_dev = np.sqrt(2.0 / fan_in)
                    kernel.array = np.random.randn(*kernel.weights.shape) * std_dev

    def forward(self, image):
        self.layers[0].inputs.append(image)
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                layer.convolve()
            elif isinstance(layer, PoolLayer):
                layer.pool()
            elif isinstance(layer, FlatLayer):
                layer.flatten()
            else:
                raise TypeError(f'Invalid or Incomplete layer definition: {layer} of type {type(layer)}')

        self.classifier.forward(self.layers[-1].output[0])

    def backward(self, label):
        pass



