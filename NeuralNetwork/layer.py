"""
Layer

This module contains the Layer class.

Author: Julien THOMAS
Date: June 02, 2023
"""

from NeuralNetwork.neuron import Neuron
from NeuralNetwork.activation import *


class Layer:
    """
    A Layer is a group of neurons (Neuron).

    Layers are fully connected and all neurons of a same layer have the same activation function.
    The input layer have no activation and no inputs.

    Attributes:
        neurons (List[Neuron]): array of neurons that compose the layer.
        activation (str): lowercase name of the neurons' activation function.

    Methods:
        forward(inputs): input feed forward at layer level.
        backward(prev_layer, grad_wrt_outputs, grads_w, grads_b): gradient backpropagation at layer level.
        get_output(): getter on the output of the neurons.
        get_weight(): getter on the weights of the neurons.
    """
    def __init__(self, size=0, activation=None):
        """
        Initializes a new instance of Layer.

        Args:
            size (int): number of neurons in the layer.
            activation (str): lowercase name of the activation function (default is identity).
        """
        self.neurons = [Neuron(activation) for _ in range(size)]
        if activation == 'relu':
            self.activation = relu
        else:
            self.activation = identity

    def __len__(self):
        return len(self.neurons)

    def forward(self, inputs):
        """
        Sets neurons' input and then calls the activation function.

        Args:
            inputs (list or np.ndarray): output values of neurons in the previous layer.

        Returns:
            None
        """
        for neuron in self.neurons:
            neuron.inputs = inputs
            neuron.activate()

    def backward(self, prev_layer, grad_wrt_outputs, grads_w, grads_b):
        """
        Computes the derivatives of the outputs of the layer, then the gradient of the loss w.r.t to these values
        to finally get the gradient of the loss w.r.t weights and biases of the neurons in the layer. Arrays from the
        network level backward pass are updated here each time.

        Args:
            prev_layer (Layer): layer on the right.
            grad_wrt_outputs (list or np.ndarray): gradients w.r.t outputs of the previous layer.
            grads_w (list or np.ndarray): array storing gradients w.r.t weights for every layer.
            grads_b (list or np.ndarray): array storing gradients w.r.t biases for every layer.

        Returns:
            grad_wrt_outputs (list or np.ndarray): new gradients w.r.t outputs.
        """
        act_derivatives = [self.activation(n.output, derivative=True) for n in self.neurons]
        grad_wrt_activations = np.dot(grad_wrt_outputs, prev_layer.get_weights()) * act_derivatives

        # Compute gradient of weights and biases for the current layer
        grad_wrt_weights = np.outer(grad_wrt_activations, self.get_input())
        grad_wrt_biases = grad_wrt_activations

        grads_w.insert(0, grad_wrt_weights)
        grads_b.insert(0, grad_wrt_biases)
        # activations are the output of the layer on the left (next iteration)
        grad_wrt_outputs = grad_wrt_activations
        return grad_wrt_outputs

    def get_output(self):
        """
        Getter on the output of the layer (e.g. all the neurons' activation values)

        Returns:
            list: array of the layer's output
        """
        return [n.output for n in self.neurons]

    def get_input(self):
        """
        Getter on the input of the layer (e.g. all the previous layer neurons' activation values)

        Returns:
            list: array of the layer's input
        """
        return self.neurons[0].inputs

    def get_weights(self):
        """
        Getter on the weights of the layer (e.g. all the weighting factors on the layer's input)

        Returns:
            list: array of the layer's weights
        """
        return [n.weights for n in self.neurons]


