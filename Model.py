import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.bias    = np.zeros(output_size)
        # for backprop
        self.input = None
        self.output = None

    def forward(self, X):
        # X: (batch_size, input_size) or (input_size,)
        self.input = X
        return X @ self.weights.T + self.bias

    def activation(self, Z):
        return np.tanh(Z)

    def activation_deriv(self, A):
        return 1 - A**2

    def backward(self, d_output):
        # d_output: same shape as self.output
        # for batch, shapes: (batch_size, output_size)
        if self.input.ndim == 1:
            d_weights = np.outer(d_output, self.input)
        else:
            d_weights = d_output.T @ self.input  # (output_size, input_size)
        d_bias = d_output.mean(axis=0) if d_output.ndim > 1 else d_output
        d_input = d_output @ self.weights     # (batch_size, input_size)
        return d_input, d_weights, d_bias

class Model:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.fitness = None

    def process_batch(self, X):
        A = X
        self._activations = [A]
        for layer in self.layers:
            Z = layer.forward(A)
            A = layer.activation(Z)
            self._activations.append(A)
        return A

    def backprop_batch(self, X, Y, lr=1e-3):
        # X: (batch, in), Y: (batch,) or (batch, out)
        A = self.process_batch(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        # derivative of MSE and tanh
        delta = (A - Y) * (1 - A**2)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self._activations[i]
            d_input, d_w, d_b = layer.backward(delta)
            # update
            layer.weights -= lr * d_w / X.shape[0]
            layer.bias    -= lr * d_b / X.shape[0]
            delta = d_input * (1 - A_prev**2)

    def copy(self):
        new = Model([l.input_size for l in self.layers] + [self.layers[-1].output_size])
        for src, dst in zip(self.layers, new.layers):
            dst.weights = np.copy(src.weights)
            dst.bias    = np.copy(src.bias)
        return new

    def crossover(self, other):
        child = self.copy()
        for i, layer in enumerate(child.layers):
            other_layer = other.layers[i]
            # only if shapes match
            if layer.weights.shape == other_layer.weights.shape:
                mask = np.random.rand(*layer.weights.shape) < 0.5
                layer.weights = np.where(mask, layer.weights, other_layer.weights)
            if layer.bias.shape == other_layer.bias.shape:
                mask = np.random.rand(*layer.bias.shape) < 0.5
                layer.bias = np.where(mask, layer.bias, other_layer.bias)
        return child
