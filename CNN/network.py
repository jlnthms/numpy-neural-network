from typing import List

from CNN.layer import *
from NeuralNetwork.network import MultiLayerPerceptron


class ConvNeuralNet:
    def __init__(self, layers: List[Layer], classifier: MultiLayerPerceptron):
        self.layers = layers
        self.classifier = classifier

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



