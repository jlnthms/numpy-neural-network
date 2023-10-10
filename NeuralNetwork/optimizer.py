"""
Optimizer

This module contains the Optimizer class that is instanced during
the training of MultiLayerPerceptron.

Author: Julien THOMAS
Date: September 22, 2023
"""

from NeuralNetwork.network import *


class Optimizer:
    """
    An optimizer class for training a Multi-Layer Perceptron (MLP).

    This optimizer is responsible for updating the weights and biases of the MLP
    during the training process using a specific optimization algorithm, such as
    gradient descent or its variants (e.g., Adam, RMSprop).

    Attributes:
        learning_rate (float): learning rate for training.

    Methods:
        step(layers, grads_w, grads_b): Updates weights and biases according to the chosen algorithm.

    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers: List[Layer], grads_w, grads_b):
        pass


class SGD(Optimizer):
    """
    Derived class of Optimizer for Stochastic Gradient Descent (SGD) optimization.

    SGD aims to estimate the "direction" towards minimum training loss using its derivative,
    and update weights and biases in consequence.
    """
    def __init__(self, learning_rate, momentum=None):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def step(self, layers: List[Layer], grads_w, grads_b):
        for layer, grad_w, grad_b in zip(layers[1:], grads_w, grads_b):
            for neuron, grad_weights, grad_bias in zip(layer.neurons, grad_w, grad_b):
                # Check if the neuron has been encountered before, and initialize its velocity if not
                if neuron not in self.velocities:
                    self.velocities[neuron] = {
                        'weights': [0.0] * len(neuron.weights),
                        'bias': 0.0
                    }
                if self.momentum is not None:
                    # Update weights for each neuron in the layer
                    for i in range(len(neuron.weights)):
                        velocity = self.momentum * self.velocities[neuron]['weights'][i] + grad_weights[i]
                        neuron.weights[i] -= self.learning_rate * velocity
                        self.velocities[neuron]['weights'][i] = velocity

                        # Update biases for each neuron in the layer with momentum
                        velocity = self.momentum * self.velocities[neuron]['bias'] + grad_bias
                        neuron.bias -= self.learning_rate * velocity
                        self.velocities[neuron]['bias'] = velocity
                else:
                    # If momentum is not provided, perform regular SGD
                    for i in range(len(neuron.weights)):
                        neuron.weights[i] -= self.learning_rate * grad_weights[i]
                    neuron.bias -= self.learning_rate * grad_bias


