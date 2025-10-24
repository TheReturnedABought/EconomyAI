import glob
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Model import Model as m
from Model import get_device as gd
from Model import save_model as sm
from Model import load_model as lm
from trading_phases import train
from trading_phases import TradingSimulation
import os

# ================= Buy-and-Hold Strategy =================
class BuyAndHold:
    def __init__(self, initial_cash=100.0, stock_name="UNKNOWN"):
        self.initial_cash = initial_cash
        self.stock_name = stock_name
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.buy_price = 0
        self.portfolio_value = [self.initial_cash]
        self.buy_executed = False
        self.history = []

    def execute_strategy(self, price, step=None):
        if not self.buy_executed and self.cash > 0:
            self.shares = self.cash / price
            self.cash = 0
            self.buy_price = price
            self.buy_executed = True

        total_value = self.cash + self.shares * price
        self.portfolio_value.append(total_value)
        self.history.append({
            'step': step,
            'cash': self.cash,
            'shares': self.shares,
            'total_value': total_value
        })
        return total_value

    def get_current_value(self, price):
        return self.cash + self.shares * price

    def get_return(self):
        if self.portfolio_value:
            return (self.portfolio_value[-1] / self.initial_cash - 1) * 100
        return 0
# ================= Main =================
if __name__ == "__main__":
    data_files = glob.glob('./DATA/*.csv')
    print(f"Found {len(data_files)} data files.")

    if not data_files:
        print("No data files found!")
        exit()

    input_size = 6
    output_size = 3
    model = m(input_size, output_size=output_size)
    model_path = 'trained_trading_model.pth'

    # --- Load existing model if present ---
    if os.path.exists(model_path):
        print(f"🔄 Loading existing model from '{model_path}'...")
        lm(model, model_path)
    else:
        print("🏗️ No existing model found. Preparing to train a new one...")

    # --- Ask user for phase selection ---
    print("\nSelect training phase:")
    print("0: Phase 0 (Buy-and-Hold pre-training)")
    print("1: Phase 1 Buy Low, Sell High)")
    print("2: Phase 2 (all stocks, end-of-stock reward)")
    print("3: Auto (Phase 1 → Phase 2)")
    choice = input("Enter 0, 1, 2, or 3: ").strip()
    phase_map = {"0": "phase0", "1": "phase1", "2": "phase2", "3": "auto"}
    selected_phase = phase_map.get(choice, "auto")
    print(f"\nSelected training mode: {selected_phase}\n")

    # --- Buy-and-Hold Baseline ---
    #print("\n🏁 Running Buy-and-Hold baseline for all stocks...")
    #bh_returns = {}
    #for file in data_files:
    #    df = pd.read_csv(file)
    #    stock_name = os.path.basename(file).split('.')[0]

    #    bh = BuyAndHold(stock_name=stock_name)
    #    for i in range(len(df)):
    #        if 'Close' not in df.columns or pd.isna(df.iloc[i]['Close']):
    #            continue
    #        bh.execute_strategy(df.iloc[i]['Close'], step=i)

    #    stock_return = bh.get_return()
    #    value_percent = (bh.get_current_value(df.iloc[-1]['Close']) / bh.initial_cash) * 100
    #    bh_returns[stock_name] = stock_return
    #    print(f"{stock_name}: Return={stock_return:+.2f}% | Value={value_percent:.2f}%")

    #avg_bh = np.mean(list(bh_returns.values()))
    #print(f"\nAverage Buy-and-Hold Return Across All Stocks: {avg_bh:+.2f}%")
    print("Device in use:", gd())

    # --- Train Model based on selected phase ---
    trained_model = train(model, data_files, phase=selected_phase)
    sm(trained_model, model_path)
    print(f"\n💾 Model saved as '{model_path}'")