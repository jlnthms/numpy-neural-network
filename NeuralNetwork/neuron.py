"""
Neuron

This module contains the Neuron class which is the primary unit
of the neural network.

Author: Julien THOMAS
Date: September 22, 2023
"""

from NeuralNetwork.activation import *


class Neuron:
    """
    The neuron is the fundamental unit of the artificial neural network.

    Neurons are grouped in layers, and each neuron is connected to all neurons of both the previous and
    following layer. Connections are made through weights that allow to perform successive linear combinations
    of the input. When a neuron activates it applies non-linearity on these combinations to introduce the capability
    to capture more complex features and relationships within the data than a simple weighted sum.

    Attributes:
        inputs (list or np.ndarray):
        weights (list or np.ndarray):
        bias (float):
        activation (str):

    Methods:
        activate():
    """
    def __init__(self, activation: str):
        """
        Initializes a new instance of Neuron.

        Args:
            activation (str): lowercase name of the activation function

        Returns:
            None
        """
        self.inputs = []
        self.weights = []
        self.bias = 0.0
        self.is_active = True
        if activation == 'relu':
            self.activation = relu
        else:
            self.activation = identity

        self.output = 0.0

    def activate(self):
        """
        Computes the weighted sum of inputs and calls the designated activation.

        Returns:
            None
        """
        weighted_sum = sum(self.weights[i] * self.inputs[i] for i in range(len(self.inputs)))
        self.output = self.activation(weighted_sum + self.bias)
