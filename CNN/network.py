"""
Network

This module contains the main class of the project: ConvNeuralNet.
A network is composed Layer objects, themselves composed of Kernel objects.

Author: Julien THOMAS
Date: November 04, 2023
"""

from typing import List

from CNN.layer import *
from NeuralNetwork.network import MultiLayerPerceptron


class ConvNeuralNet:
    """
        A Convolutional Neural Network (CNN) implementation.

        The Convolutional Neural Network is a feedforward supervised network
        for problems involving image data. It is composed of Convolution layers for feature
        extraction, Pooling players for dimensionality reduction and of an MLP classifier
        (or dense layer(s) to perform a prediction).

        Attributes:
            layers (List[Layer]): list of layers from input to output.
            classifier (MultiLayerPerceptron): Dense prediction module.

        Methods:
            he_initialization():
                Defines a range for random values initialization of kernels to prevent
                the infamous vanishing/exploding gradient problem.

            forward(X):
                Forward pass of the pixels through the network.

            backward(y):
                Backward propagation of gradients w.r.t kernels using the calculus chain rule.
        """
    def __init__(self, layers: List[Layer], classifier: MultiLayerPerceptron):
        self.layers = layers
        # TODO: the argument should only be a list of Dense layers (from NeuralNetwork) and the classifier is created
        #  in the constructor, input size is computed from the architecture size reduction and output size from label
        #  size.
        self.classifier = classifier

    def he_initialization(self):
        """
        He Initialization (He et al.(2015): https://arxiv.org/abs/1502.01852).
        Filtering pixels for all kernels in a layer are set to sqrt(2 / incoming kernels) * r
        with r: a random matrix with values ranging from 0 to the size of the kernel.

        The method calls the he_init method from NeuralNetwork to initialize weights from
        the prediction module of the CNN.

        Returns:
            None
        """
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
        """
        Forward pass of the CNN. Each input image map is passed through the layers.
        Depending on the layer type, the appropriate class method is applied (see CNN/Layer).

        Args:
            image (numpy.ndarray): input image to be passed forward.

        Returns:
            None
        """
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
        """
        Backward propagation of the gradient of the loss w.r.t kernels. The backward pass of the
        classifier is called first (see NeuralNetwork/network). The final gradient array of the dense
        layers must be unflatten to be match the input tensor shape of the flattening layer. Layer backward methods
        are called depending on layer type.

        Args:
            label (list or numpy.ndarray): label(s) of the input(s) just passed forward.

        Returns:
            tuple: (numpy.ndarray) gradients w.r.t weights/biases/kernels for each layer
        """
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
