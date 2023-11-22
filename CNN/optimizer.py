from typing import List

from CNN.layer import *


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers: List[Layer], grads_k):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        pass

    def step(self, layers: List[Layer], grads_k):
        for layer, kernel_grads in zip(layers, grads_k):
            for kernel, kernel_grad in zip(layer.kernels, kernel_grads):
                kernel.array -= self.learning_rate * kernel_grad
