"""
Optimizer

This module contains the Optimizer class that is instanced during
the training of ConvNeuralNet.

Author: Julien THOMAS
Date: November 04, 2023
"""

from typing import List

from CNN.layer import *
from NeuralNetwork.optimizer import SGD as NnSGD


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, classifier, layers: List[Layer], grad_class_w, grad_class_b, grads_k):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.classifier_optimizer = NnSGD(learning_rate)

    def step(self, classifier, layers: List[Layer], grad_class_w, grad_class_b, grads_k):
        self.classifier_optimizer.step(classifier.layers, grad_class_w, grad_class_b)
        for layer, kernel_grads in zip(layers, grads_k):
            if isinstance(layer, ConvLayer):
                for kernel, kernel_grad in zip(layer.kernels, kernel_grads):
                    kernel.array -= self.learning_rate * kernel_grad
