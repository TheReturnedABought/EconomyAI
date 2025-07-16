# Model.py
import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.random.randn(output_size) * 0.1
        self.input = None
        self.output = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weights, x) + self.bias
        return self.output

    def backward(self, d_output, learning_rate=0.01):
        d_input = np.dot(self.weights.T, d_output)
        self.d_weights = np.outer(d_output, self.input)
        self.d_bias = d_output
        return d_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


class Model:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def process(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backpropagate(self, x, target, learning_rate=0.01):
        # Forward pass
        output = self.process(x)

        # Compute loss derivative (MSE)
        error = output - target
        d_loss = 2 * error

        # Backward pass
        d = d_loss
        for layer in reversed(self.layers):
            d = layer.backward(d)

        # Update weights
        for layer in self.layers:
            layer.update(learning_rate)

    def copy(self):
        new_model = Model([l.input_size for l in self.layers] + [self.layers[-1].output_size])
        for i, layer in enumerate(self.layers):
            new_model.layers[i].weights = layer.weights.copy()
            new_model.layers[i].bias = layer.bias.copy()
        return new_model