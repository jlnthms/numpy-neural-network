"""
Network

This module contains the main class of the project: MultiLayerPerceptron.
A network is composed Layer objects, themselves composed of Neuron objects.

Author: Julien THOMAS
Date: June 02, 2023
"""

from typing import List

from NeuralNetwork.layer import Layer
from NeuralNetwork.loss import *


class MultiLayerPerceptron:
    """
    A Multi-Layer Perceptron (MLP) neural network implementation.

    The Multi-Layer Perceptron is a feedforward artificial neural network
    comprised of multiple layers, including input, hidden, and output layers.
    Each layer contains neurons that process and propagate data.

    Attributes:
        layers (List[Layer]): list of layers from input to output.

    Methods:
        he_init():
            Defines a range for random values initialization of weights to prevent
            the infamous vanishing/exploding gradient problem.

        forward(X):
            Forward pass of the input through the network.

        backward(y):
            Backward propagation of gradients w.r.t weights using the calculus chain rule.
    """
    def __init__(self, layers: List[Layer]):
        """
        Initializes a new instance of MultiLayerPerceptron.

        Args:
            layers (List[Layer]): list of layers from input to output

        Returns:
            None
        """
        self.layers = layers

    def xavier_init(self):
        pass

    def he_init(self):
        """
        He Initialization (He et al.(2015): https://arxiv.org/abs/1502.01852).
        Weights for all neurons in a layer are set to sqrt(2 / incoming neurons) * r
        with r: a random value ranging from 0 to the size of the previous layer.

        Returns:
            None
        """
        for i in range(1, len(self.layers), 1):
            for neuron in self.layers[i].neurons:
                size_prev = len(self.layers[i - 1])
                neuron.weights = list(np.random.randn(size_prev) * np.sqrt(2 / size_prev))

    def forward(self, X):
        """
        Forward pass of the neural network. Neurons of each layer get the outputs
        of the previous layer for input, and they activate. The output of the network
        after passing forward an input is an array of (yet) un-normalized scores.

        Args:
            X (list or numpy.ndarray): array of inputs

        Returns:
            None
        """
        input_layer = self.layers[0]
        # Assign each input neuron value to X values
        for i in range(len(input_layer)):
            input_layer.neurons[i].output = X[i]
        # Activate each neuron of each layer from left to right (except input of course)
        for i, layer in enumerate(self.layers[1:], 1):
            layer.forward([n.output for n in self.layers[i - 1].neurons])

    def backward(self, y):
        """
        Backward propagation of the gradient of the loss w.r.t weights. The gradients
        for the weights of each layer are computed using the weights, activations and gradients
        from the previous layer.

        Args:
            y (list or numpy.ndarray): label(s) of the input(s) just passed forward.

        Returns:
            tuple: (numpy.ndarray) gradients w.r.t weights/biases for each layer
        """
        # TODO: change with layer.activation()/loss to be flexible
        # compute the gradient of loss with respect to the un-normalized scores of the output layer
        output = self.layers[-1].get_output()
        grad_wrt_outputs = np.dot(softmax(output, derivative=True), cross_entropy(output, y, derivative=True))
        # apply the chain rule to calculate the gradients w.r.t weights
        grad_wrt_weights = np.outer(grad_wrt_outputs.T, self.layers[-2].get_output())
        grads_w, grads_b = [grad_wrt_weights], []
        # Back-propagate gradients through hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            # Compute gradient of activations for the current layer
            curr_layer, prev_layer = self.layers[i], self.layers[i+1]
            curr_layer.backward(prev_layer, grad_wrt_outputs, grads_w, grads_b)

        return grads_w, grads_b, grad_wrt_outputs

    def print(self):
        pass
