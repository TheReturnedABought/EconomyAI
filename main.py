import os
import numpy as np
import pandas as pd
from random import random, randint, choice
from tqdm import tqdm
from Model import Model

class Layer:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.bias    = np.zeros(output_size)
        self.input_size  = input_size
        self.output_size = output_size

    def forward(self, X):
        return X @ self.weights.T + self.bias

    def activation(self, Z):
        return np.tanh(Z)

class Generation:
    def __init__(self, num_models, layer_sizes):
        self.models = [Model(layer_sizes) for _ in range(num_models)]

    def evaluate_fitness(self, X, Y):
        outputs = np.stack([m.process_batch(X).flatten() for m in self.models])
        mses = ((outputs - Y[None,:])**2).mean(axis=1)
        # Compute complexity on the fly (sum of weights+bias elements per model)
        complexities = np.array([
            sum(layer.weights.size + layer.bias.size for layer in m.layers)
            for m in self.models
        ])
        self.fitnesses = -mses - 1e-4 * complexities
        for m, f in zip(self.models, self.fitnesses):
            m.fitness = f


    def selection_multiply(self):
        idx = np.argsort(self.fitnesses)[::-1]
        top2 = idx[:2]
        num_replace = len(self.models) // 2  # replace half population
        new_models = [self.models[i] for i in top2]  # keep best two
        # generate children via crossover
        for _ in range(num_replace):
            parent1, parent2 = choice(top2), choice(top2)
            child = self.models[parent1].crossover(self.models[parent2])
            new_models.append(child)
        # fill rest by elitism
        while len(new_models) < len(self.models):
            new_models.append(self.models[top2[0]].copy())
        self.models = new_models

    def mutate(self, base_rate):
        # mutate all except the very best
        for m in self.models[1:]:
            for li in range(len(m.layers)):
                if random() < base_rate:
                    self._mutate_layer(m, li)
            m.complexity = sum(l.weights.size + l.bias.size for l in m.layers)

    def _mutate_layer(self, model, layer_idx):
        layer = model.layers[layer_idx]
        if layer_idx < len(model.layers)-1:
            next_layer = model.layers[layer_idx+1]
            if random() < 0.5 and next_layer is not model.layers[-1]:
                self._add_neuron(layer, next_layer)
            elif layer.output_size > 1:
                self._remove_neuron(layer, next_layer)
        else:
            # for last layer, just perturb weights/bias
            layer.weights += np.random.randn(*layer.weights.shape) * 0.01
            layer.bias    += np.random.randn(layer.bias.size) * 0.01

    def _add_neuron(self, layer, next_layer):
        new_w = np.random.randn(1, layer.input_size)*0.1
        layer.weights = np.vstack([layer.weights, new_w])
        layer.bias    = np.append(layer.bias, np.random.randn()*0.1)
        layer.output_size += 1
        new_c = np.random.randn(next_layer.output_size,1)*0.1
        next_layer.weights = np.hstack([next_layer.weights, new_c])
        next_layer.input_size += 1

    def _remove_neuron(self, layer, next_layer):
        idx = randint(0, layer.output_size-1)
        layer.weights = np.delete(layer.weights, idx, axis=0)
        layer.bias    = np.delete(layer.bias, idx)
        layer.output_size -= 1
        next_layer.weights = np.delete(next_layer.weights, idx, axis=1)
        next_layer.input_size -= 1

    def train_with_backprop(self, X, Y, learning_rate=1e-3):
        for m in self.models[1:]:
            m.backprop_batch(X, Y, learning_rate)


def load_stock_data(directory):
    data = []
    for fn in os.listdir(directory):
        if not fn.endswith('.xlsx'): continue
        df = pd.read_excel(os.path.join(directory, fn))
        if 'Close' not in df: continue
        closes = df['Close'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = (closes[1:] - closes[:-1]) / closes[:-1]
        data.extend(rets[np.isfinite(rets)])
    return np.array(data)


def create_sequences(data, seq_len=10):
    X, Y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.stack(X), np.array(Y)


def save_model(model, fname):
    params = {}
    for i, layer in enumerate(model.layers):
        params[f"w{i}"] = layer.weights
        params[f"b{i}"] = layer.bias
    params["sizes"] = [layer.input_size for layer in model.layers] + [model.layers[-1].output_size]
    np.savez(fname, **params)


def main():
    DATA_DIR    = "DATA"
    SEQ_LENGTH  = 10
    NUM_MODELS  = 50
    GENERATIONS = 100
    best_so_far = -100
    print("Loading data…")
    rets = load_stock_data(DATA_DIR)
    print(f"Total returns: {len(rets)}")
    if len(rets) < SEQ_LENGTH+10:
        raise RuntimeError("Not enough data!")

    X, Y = create_sequences(rets, SEQ_LENGTH)
    print(f"Prepared {X.shape[0]} samples.")

    gen = Generation(NUM_MODELS, layer_sizes=[SEQ_LENGTH, 8, 4, 1])
    for g in tqdm(range(GENERATIONS), desc="Evolving"):
        # 1) Evaluate current fitness
        gen.evaluate_fitness(X, Y)
        current_best = gen.fitnesses.max()

        # 2) Check for improvement
        if current_best > best_so_far:
            best_so_far = current_best
            no_improve  = 0
        else:
            no_improve += 1
        rate = 0.1 if no_improve < 5 else no_improve**0.5
        gen.selection_multiply()
        gen.mutate(base_rate=0.2)  # increased mutation rate
        gen.train_with_backprop(X, Y, learning_rate=1e-3)
        if g % 10 == 0:
            tqdm.write(f"Generation {g}: Best fitness = {gen.fitnesses.max():.4f}")
    gen.evaluate_fitness(X, Y)

    best_model = max(gen.models, key=lambda m: m.fitness)
    save_model(best_model, "stock_predictor.npz")
    print("Done! Best model saved to stock_predictor.npz")

if __name__ == "__main__":
    main()
