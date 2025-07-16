import os
import numpy as np
import pandas as pd
from random import random, randint, choice
from tqdm import tqdm
from Model import Model

# Trading GA that maximizes cash generated in backtest
class Generation:
    def __init__(self, num_models, layer_sizes, seq_length):
        self.seq_length = seq_length
        self.models = [Model(layer_sizes) for _ in range(num_models)]

    def evaluate_fitness(self, X, prices):
        # X: (T, seq_len, feature_dim)
        # prices: (T,)
        fitnesses = []
        for m in self.models:
            cash = 1000.0
            pos = 0
            for t in range(X.shape[0]):
                # get features flattened for batch processing
                seq = X[t].flatten()[None, :]
                signal = m.process_batch(seq)[0,0]
                price = prices[t]
                # buy
                if signal > 0 and cash >= price:
                    cash -= price
                    pos += 1
                # sell
                elif signal < 0 and pos > 0:
                    cash += price
                    pos -= 1
            # liquidate remaining
            fitnesses.append(cash + pos * prices[-1])
        self.fitnesses = np.array(fitnesses)
        for m, f in zip(self.models, self.fitnesses):
            m.fitness = f

    def selection_multiply(self):
        idx = np.argsort(self.fitnesses)[::-1]
        top2 = idx[:2]
        pop = [self.models[i].copy() for i in top2]
        # produce rest via crossover
        while len(pop) < len(self.models):
            p1, p2 = choice(top2), choice(top2)
            pop.append(self.models[p1].crossover(self.models[p2]))
        self.models = pop

    def mutate(self, base_rate):
        for m in self.models[1:]:
            for li in range(len(m.layers)):
                if random() < base_rate:
                    m._mutate_layer(li)

    def backtest_train(self, X, prices, generations, lr, base_rate):
        for g in tqdm(range(generations), desc="Evolving"):
            self.evaluate_fitness(X, prices)
            self.selection_multiply()
            self.mutate(base_rate)
        # final eval
        self.evaluate_fitness(X, prices)


def load_and_preprocess(directory):
    dfs = []
    for fn in os.listdir(directory):
        if not fn.endswith('.xlsx') or fn.startswith('~$'):
            continue
        full_path = os.path.join(directory, fn)
        try:
            df = pd.read_excel(full_path)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.dropna(subset=['Close', 'DIFof2Av'])
            df['Close'] = df['Close'].replace(r'[\$,]', '', regex=True).astype(float)
            df['DIFof2Av'] = pd.to_numeric(df['DIFof2Av'].astype(str).str.rstrip('%'), errors='coerce') / 100.0
            dfs.append(df[['Close', '5DayMAV', '20DayMAV', 'DIFof2Av']])
        except Exception as e:
            print(f"[WARN] Skipped file {fn} due to error: {e}")
            continue

    if not dfs:
        raise RuntimeError("No valid Excel files loaded from directory.")

    data = pd.concat(dfs).reset_index(drop=True)
    # normalize
    data = (data - data.mean()) / data.std()
    return data.values


def create_sequences(data, seq_len):
    X, prices = [], []
    # data shape (N, features)
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        prices.append(data[i + seq_len, 0])
    return np.stack(X), np.array(prices)


def main():
    DATA_DIR = 'DATA'
    SEQ_LENGTH = 10
    NUM_MODELS = 50
    GENERATIONS = 10
    BASE_RATE = 0.1
    LEARNING_RATE = 1e-3  # unused here

    raw = load_and_preprocess(DATA_DIR)
    X, prices = create_sequences(raw, SEQ_LENGTH)

    feature_dim = raw.shape[1]
    layer_sizes = [SEQ_LENGTH * feature_dim, 8, 4, 1]

    gen = Generation(NUM_MODELS, layer_sizes, SEQ_LENGTH)
    gen.backtest_train(X, prices, GENERATIONS, LEARNING_RATE, BASE_RATE)

    # Save best
    best = max(gen.models, key=lambda m: m.fitness)
    np.savez('best_trader.npz', fitness=best.fitness)
    print(f"Best cash: {best.fitness}")


if __name__ == '__main__':
    main()
