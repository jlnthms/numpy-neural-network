# NumPy Neural Network

A Python implementation of a multi-layer perceptron (MLP) neural network using NumPy.

![Neural Network](https://github.com/jlnthms/numpy-neural-network/assets/74052135/0cac744a-4742-4eb1-8f92-0c183acb0987)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Upcoming Extensions](#upcoming-extensions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Author](#author)

## Introduction

This project provides an educational implementation of a multi-layer perceptron (MLP) neural network using NumPy. 
The goal is to delve deep into the theory of neural networks by representing layers and neurons as objects, making 
it a clear and insightful learning tool. Similar projects tend to represent layers directly as arrays, which is very 
efficient and simple, yet it does not provide a very clear and representative structure of the neural network from an object perspective.
While primarily designed for personal learning, it can also serve as a foundation 
for building more complex neural networks as the code was designed to be flexible.

## Key Features

- **Layer and Neuron Objects:** Layers and neurons are represented as objects, allowing for a more intuitive understanding of neural network structure.
- **Flexible structure:** New activations/loss/optimizations can be added and work straight away.
- **Above and beyond:** Further extensions towards more advanced concepts of neural networks are currently under consideration.

## Upcoming Extensions

- **Mini-batch backpropagation:** Only instance-based training is currently available.
- **Adam Optimization:** Only SGD is implemented at the moment.
- **Possibility to save a model:** weights can be saved to a file to run some experiments without retraining.
- **Dropout regularization:** Randomly deleting connections between neurons help reducing redundancy and improve robustness.

## Getting Started

### Prerequisites

To run this project, you'll need:

- Python (3.7+ recommended)
- NumPy
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

The notebook *test.ipynb* already provides a concrete runnable use case of handwritten digits 
classification, however here is a more generic example:
   
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

## Author

**Julien Thomas**
- GitHub: [JulienThomas](https://github.com/jlnthms)
- Email: thomasjulien92140@gmail.com
- LinkedIn: https://www.linkedin.com/in/julien-thomas-826b4920b/
