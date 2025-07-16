import os
import pandas as pd
import numpy as np
from random import random
from tqdm import tqdm
from Model import Model


class Generation:
    def __init__(self, num_models, layer_sizes):
        self.models = [Model(layer_sizes) for _ in range(num_models)]

    def evaluate_fitness(self, input_data, target_outputs):
        for model in self.models:
            total_error = 0
            for x, target in zip(input_data, target_outputs):
                output = model.process(x)
                # Convert output to scalar if it's an array
                if isinstance(output, np.ndarray):
                    output = output.item()  # Convert 1-element array to scalar
                total_error += (output - target) ** 2  # MSE
            complexity = sum(l.weights.size + l.bias.size for l in model.layers)
            model.fitness = -total_error - 0.0001 * complexity

    def selection_multiply(self):
        self.models.sort(key=lambda m: m.fitness, reverse=True)
        num_to_replace = len(self.models) // 4
        for i in range(-num_to_replace, 0):
            if random() < 0.2:
                self.models[i] = self.models[1].copy()
            else:
                self.models[i] = self.models[0].copy()

    def mutate(self, base_rate):
        for model in self.models[10:]:
            for layer_idx in range(1, len(model.layers) - 2):
                if random() < base_rate:
                    self._mutate_layer(model, layer_idx)

    def _mutate_layer(self, model, layer_idx):
        layer = model.layers[layer_idx]
        next_layer = model.layers[layer_idx + 1]

        if random() < 0.25 and next_layer != model.layers[-1]:
            self._add_neuron(layer, next_layer)
        elif random() < 0.25 and layer.output_size > 1 and next_layer != model.layers[-1]:
            self._remove_neuron(layer, next_layer)

    def _add_neuron(self, layer, next_layer):
        new_weights = np.random.randn(1, layer.input_size) * 0.1
        layer.weights = np.vstack([layer.weights, new_weights])
        layer.bias = np.append(layer.bias, np.random.randn(1) * 0.1)
        layer.output_size += 1

        new_col = np.random.randn(next_layer.output_size, 1) * 0.1
        next_layer.weights = np.hstack([next_layer.weights, new_col])
        next_layer.input_size += 1

    def _remove_neuron(self, layer, next_layer, neuron_idx=None):
        if neuron_idx is None:
            neuron_idx = np.random.randint(0, layer.output_size)
        layer.weights = np.delete(layer.weights, neuron_idx, axis=0)
        layer.bias = np.delete(layer.bias, neuron_idx)
        layer.output_size -= 1
        next_layer.weights = np.delete(next_layer.weights, neuron_idx, axis=1)
        next_layer.input_size -= 1

    def train_with_backprop(self, input_data, target_outputs, learning_rate=0.01):
        for model in self.models[1:]:
            for x, target in zip(input_data, target_outputs):
                model.backpropagate(x, target, learning_rate)


def load_stock_data(directory):
    """Load and preprocess stock data from Excel files"""
    data = []
    for file in os.listdir(directory):
        if file.endswith('.xlsx'):
            try:
                filepath = os.path.join(directory, file)
                df = pd.read_excel(filepath)
                if 'Close' in df.columns:
                    closes = df['Close'].values
                    # Calculate returns and handle division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        returns = (closes[1:] - closes[:-1]) / closes[:-1]
                    # Remove NaNs and Infs
                    valid_returns = returns[np.isfinite(returns)]
                    data.extend(valid_returns)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    return np.array(data)


def create_sequences(data, seq_length=10):
    """Create input sequences and targets"""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - 1):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])  # Target is scalar
    return np.array(sequences), np.array(targets)


def save_model(model, filename):
    model_data = {}
    for i, layer in enumerate(model.layers):
        model_data[f"layer_{i}_weights"] = layer.weights
        model_data[f"layer_{i}_biases"] = layer.bias
    model_data["layer_sizes"] = [layer.input_size for layer in model.layers] + [model.layers[-1].output_size]
    np.savez(filename, **model_data)


# Configuration
DATA_DIR = "DATA"
SEQ_LENGTH = 10
NUM_MODELS = 50
GENERATIONS = 100
LAYER_SIZES = [SEQ_LENGTH, 8, 4, 1]


def main():
    # Load and prepare data
    print("Loading stock data...")
    returns = load_stock_data(DATA_DIR)
    print(f"Loaded {len(returns)} data points")

    # Handle case with insufficient data
    if len(returns) < SEQ_LENGTH + 10:
        print(f"Error: Only {len(returns)} data points. Need at least {SEQ_LENGTH + 10} to train.")
        return

    inputs, targets = create_sequences(returns, SEQ_LENGTH)
    print(f"Created {len(inputs)} training sequences")

    # Initialize generation
    gen = Generation(NUM_MODELS, LAYER_SIZES)

    # Evolution loop with progress bar
    print("Starting training...")
    for generation in tqdm(range(GENERATIONS)):
        # Evaluate fitness
        gen.evaluate_fitness(inputs, targets)

        # Track best fitness
        current_best = max(m.fitness for m in gen.models)
        print(f"Generation {generation}: Best Fitness = {current_best:.4f}")

        # Evolutionary operations
        gen.selection_multiply()
        gen.mutate(0.1)
        gen.train_with_backprop(inputs, targets, 0.001)

    # Save best model
    gen.models.sort(key=lambda m: m.fitness, reverse=True)
    best_model = gen.models[0]
    save_model(best_model, "stock_predictor.npz")
    print("Training complete. Best model saved as 'stock_predictor.npz'")


if __name__ == "__main__":
    main()