# NumPy Neural Network

A Python implementation of a multi-layer perceptron (MLP) neural network, and convolutional neural network (CNN) using NumPy.

![Neural Network](https://github.com/jlnthms/numpy-neural-network/assets/74052135/5e20644a-bb2c-4697-99c9-335a154e75b8)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Upcoming Extensions](#upcoming-extensions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Author](#author)

## Introduction

This project provides an educational implementation of a multi-layer perceptron (MLP) neural network and convolutional neural network (CNN) using NumPy. 
The goal is to delve deep into the theory of neural networks by representing layers and neurons as objects, making 
it a clear and insightful learning tool. Similar projects tend to represent layers directly as arrays, which is very 
efficient and simple, yet it does not provide a very clear and representative structure of the neural network from an object perspective.
Primarily designed for personal learning, this project serves as tool to experiment and understand neural networks following the idea of
"you don't fully understand until you made it yourself".

## Key Features

- **Layer and Neuron Objects:** Layers and neurons are represented as objects, allowing for a more intuitive understanding of neural network structure.
- **Flexible structure:** New activations/loss/optimizations can be added and work straight away.
- **Above and beyond:** Further extensions towards more advanced concepts of neural networks are currently under consideration.

## Upcoming Extensions

- **Mini-batch backpropagation:** Only instance-based training is currently available.
- **Parallelism:** CNN training is very slow and processing batches in parallel will significantly accelerate the process and allow for more complex experiments.
- **Adam Optimization:** Only SGD is implemented at the moment.
- **Possibility to save a model:** weights can be saved to a file to run some experiments without retraining.
- **Dropout regularization:** Randomly deleting connections between neurons help reducing redundancy and improve robustness.

## Getting Started

### Prerequisites

To run this project, you'll need:

- Python (3.7+ recommended)
- NumPy
- Sklearn (only to handle preprocessing)
- Jupyter (& additional dependencies)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jlnthms/numpy-neural-network.git
   cd numpy-neural-network
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Multi-Layer Perceptron

The notebook *test.ipynb* already provides a concrete runnable use case of handwritten digits 
classification with a regular neural network, however here is a more generic example:
   
```python
# Import necessary modules
import sys
sys.path.append('../')

from NeuralNetwork.network import MultiLayerPerceptron
from NeuralNetwork.loss import *
from NeuralNetwork.optimizer import SGD

# Define your network architecture
input_layer = Layer(size=input_size)
hidden_layer = Layer(size=hidden_size, activation=relu)
output_layer = Layer(size=output_size, activation=identity)

layers = [input_layer, hidden_layer, output_layer]

# Create the neural network
model = MultiLayerPerceptron(layers)

# Define your training data and labels
X_train, y_train = ...

# Training loop
num_epochs = ... # complete here
for epoch in range(num_epochs):
    # Forward pass
    model.forward(X_train)
    # Compute loss
    loss = cross_entropy(model.layers[-1].get_output(), y_train)
    # Backward pass
    grads_w, grads_b = model.backward(y_train)
    # Update weights and biases using SGD
    optimizer = SGD(learning_rate=0.01)
    optimizer.step(model.layers, grads_w, grads_b)
  ```

### Convolutional Neural Network

The CNN package has yet to be documented and extended. It is functional for instance based training on non-heavy problems
like the MNIST handwritten digit recognition. Similarly, here is an example of usage for the CNN package:

```python
# Import necessary modules
import sys
sys.path.append('../')

from CNN.network import *
from CNN.optimizer import *
from NeuralNetwork.network import *

from sklearn.preprocessing import OneHotEncoder

# Example of Yann Lecun's LeNet-5 architecture
LeNet_layers = [
    ConvLayer(6, (5, 5), 'relu'),
    PoolLayer((2, 2), stride=2, method='average'),
    ConvLayer(16, (5, 5), 'relu'),
    PoolLayer((2, 2), stride=2, method='average'),
    FlatLayer()
]

LeNet_dense_layers = [Layer(256), Layer(120, 'relu'), Layer(84, 'relu'), Layer(10, 'softmax')]
LeNet_fully_connected = MultiLayerPerceptron(LeNet_dense_layers)

LeNet_5 = ConvNeuralNet(LeNet_layers, LeNet_fully_connected)
LeNet_5.he_initialization()

# Load your data here
train_images, train_labels = ... # images must be numpy arrays

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False, categories='auto')
one_hot_encoded = encoder.fit_transform(train_labels.reshape(-1, 1))

# Training loop
epochs = ...
loss_values = []
optimizer = SGD(learning_rate=0.001)

for i in range(epochs):
    loss = 0.0
    for X, y in zip(train_images, train_labels):
        LeNet_5.forward(X)
        loss += cross_entropy(LeNet_5.classifier.layers[-1].get_output(), y)
        grad_class_w, grad_class_b, grads_k = LeNet_5.backward(y)
        optimizer.step(LeNet_5.classifier, LeNet_5.layers, grad_class_w, grad_class_b, grads_k)
    loss /= len(train_images)
    print('Epoch ', i, ' - Average loss = ', loss)
    loss_values.append(loss)

```

Note that in the notebook *visualize_activations.ipynb*, you can visualize the output feature maps for each layer of 
your model. This could serve for a debug purpose, or simply for better understanding.

## Author

**Julien Thomas**
- GitHub: [JulienThomas](https://github.com/jlnthms)
- Email: thomasjulien92140@gmail.com
- LinkedIn: https://www.linkedin.com/in/julien-thomas-826b4920b/
