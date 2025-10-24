# ================= Import Libraries =================
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from Model import Model as m
from Model import get_device as gd
import os

# ================= Trading Simulation =================
class TradingSimulation:
    """
    Simulates a trading environment for a single stock.
    Tracks:
      - Cash balance
      - Number of shares held
      - Portfolio value progression
      - List of actions taken (Sell, Hold, Buy)
    """

    def __init__(self, initial_cash=100.0):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = [self.initial_cash]
        self.actions = []

    def execute_action(self, action, price):
        trade_fraction = 0.2
        if action == 2:  # BUY
            self.shares += self.cash * trade_fraction / price
            self.cash -= self.cash * trade_fraction
        elif action == 0:  # SELL
            self.cash += self.shares * trade_fraction * price
            self.shares -= self.shares * trade_fraction

        value = self.cash + self.shares * price
        self.portfolio_value.append(value)
        self.actions.append(action)
        return value

    def get_current_value(self, price):
        return self.cash + self.shares * price

    def get_state(self, current_data):
        return np.array([
            current_data['Close'],
            current_data.get('5DayMAV', current_data['Close']),
            current_data.get('20DayMAV', current_data['Close']),
            current_data.get('DIFof2Av', 0),
            self.cash / self.initial_cash,
            (self.shares * current_data['Close']) / self.initial_cash
        ], dtype=np.float32)

# ================= Phase 0: Buy-and-Hold Pretraining =================
def train_model_phase0(model, data_files, learning_rate=1e-3, grad_clip=0.8):
    import copy

    device = gd()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_files))
    loss_fn = nn.SmoothL1Loss()
    scaler = GradScaler()  # ✅ no device needed

    ema_mean, ema_std = 0.0, 1.0
    ema_alpha = 0.05
    best_model_state = None
    best_reward = -float("inf")

    print("\n🚀 Phase 0: Buy-and-Hold Pretraining with AMP...")

    for file_path in data_files:
        df = pd.read_csv(file_path)
        stock_name = os.path.basename(file_path).split(".")[0]
        if len(df) < 30 or "Close" not in df.columns:
            continue

        first_price, last_price = df.iloc[0]["Close"], df.iloc[-1]["Close"]
        if pd.isna(first_price) or pd.isna(last_price):
            continue

        raw_reward = np.log((last_price + 1e-8) / (first_price + 1e-8))
        ema_mean = ema_alpha * raw_reward + (1 - ema_alpha) * ema_mean
        ema_std = ema_alpha * abs(raw_reward - ema_mean) + (1 - ema_alpha) * ema_std
        reward_norm = np.clip((raw_reward - ema_mean) / (ema_std + 1e-8), -3, 3)

        sim = TradingSimulation()
        sim.execute_action(2, first_price)
        for price in df["Close"].dropna().iloc[1:]:
            sim.execute_action(1, price)

        for _, row in df.iterrows():
            if pd.isna(row["Close"]):
                continue
            state = sim.get_state(row)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):  # ✅ explicit
                pred = model(state_tensor)
                target = pred.clone().detach()
                target[0, 1] = reward_norm
                target[0, 0] = reward_norm * 0.3
                target[0, 2] = reward_norm * 0.3
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        portfolio_final = sim.portfolio_value[-1]
        print(f"  {stock_name}: Reward={reward_norm:+.3f}, Final={portfolio_final:.2f}")

        if reward_norm > best_reward:
            best_reward = reward_norm
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Phase 0 complete. Best normalized reward = {best_reward:+.3f}")

    return model

# ================= Phase 1: Buy Low, Sell High =================
def train_model_phase1(
    model,
    data_files,
    num_epochs=1,
    learning_rate=5e-4,
    exploration_rate=0.05,
    min_exploration=0.005,
    grad_clip=0.8,
    alpha=0.9,   # discount factor
    beta=0.25    # reward scaling
):
    """
    Phase 1: Train the model to 'Buy Low, Sell High' using trading simulation.
    Assumes CSV columns: Date, Close, MA5, MA20, DF2MA
    """
    device = gd()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    print("\n🚀 Phase 1: Buy Low, Sell High")

    for epoch in range(num_epochs):
        total_reward = 0.0
        exploration = max(min_exploration, exploration_rate * (0.98 ** epoch))

        for file_path in data_files:
            df = pd.read_csv(file_path)
            stock_name = os.path.basename(file_path).split(".")[0]

            if len(df) < 30 or "Close" not in df.columns:
                continue

            sim = TradingSimulation()
            sim.reset()

            closes = df["Close"].values
            ma5 = df["5DayMAV"].values
            ma20 = df["20DayMAV"].values
            difs = df["DIFof2Av"].values

            for i in range(1, len(df)):
                current_data = {
                    'Close': closes[i - 1],
                    '5DayMAV': ma5[i - 1],
                    '20DayMAV': ma20[i - 1],
                    'DIFof2Av': difs[i - 1],
                }
                state = sim.get_state(current_data)

                # Predict Q-values
                q_values = model.forward(state)
                if random.random() < exploration:
                    action = random.choice([0, 1, 2])  # SELL, HOLD, BUY
                else:
                    action = torch.argmax(q_values).item()

                # Execute action
                new_value = sim.execute_action(action, closes[i])
                old_value = sim.portfolio_value[-2]
                reward = (new_value - old_value) / old_value  # relative gain/loss
                total_reward += reward

                # Compute next state
                next_data = {
                    'Close': closes[i],
                    '5DayMAV': ma5[i],
                    '20DayMAV': ma20[i],
                    'DIFof2Av': difs[i],
                }
                next_state = sim.get_state(next_data)

                # Q-learning target
                next_q = model.forward(next_state)
                target = reward + alpha * torch.max(next_q).detach()

                # Backpropagation
                optimizer.zero_grad()
                loss = loss_fn(q_values[action], target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]  |  Reward: {total_reward:.4f}  |  ε={exploration:.4f}")

    print("\n✅ Phase 1 training complete.")
    return model

# ================= Phase 2: End-of-stock Reward =================
def train_model_phase2(model, data_files, num_epochs=40, learning_rate=1e-3,
                       exploration_start=0.08, exploration_end=0.002, reward_interval=5,
                       grad_clip=0.8, alpha=1.0, beta=0.3, gamma=0.2, is_static_data=False):

    device = gd()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.SmoothL1Loss()
    scaler = GradScaler()

    best_fitness = -float("inf")
    best_model = None
    ema_reward = 0.0
    ema_alpha = 0.05

    print("\n🚀 Phase 2: Mixed Precision, Risk-Adjusted End-of-Stock Training...")

    if is_static_data:
        beta = 0.05
        gamma = 0.05
        tanh_scale = 2
        eps_static = 0.01  # small exploration to avoid flat portfolio
    else:
        tanh_scale = 5
        eps_static = None

    for epoch in range(num_epochs):
        eps = eps_static if is_static_data else max(exploration_end, exploration_start * np.exp(-0.06 * epoch))
        total_fitness, total_trades = 0.0, 0
        num_stocks = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} | ε={eps:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        for file_path in data_files:
            df = pd.read_csv(file_path)
            stock_name = os.path.basename(file_path).split(".")[0]
            if len(df) < 30 or "Close" not in df.columns:
                continue

            sim = TradingSimulation()
            portfolio_values = []
            daily_rewards = []

            # Buffers to store states, actions, rewards
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []

            for i, row in df.iterrows():
                if pd.isna(row["Close"]):
                    continue

                price = row["Close"]
                prev_val = sim.get_current_value(price)
                state = sim.get_state(row)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                # Select action
                with torch.no_grad():
                    q_vals = model(state_tensor)
                    action = torch.argmax(q_vals).item()
                if np.random.rand() < eps:
                    action = np.random.choice([0, 1, 2])

                # Execute action
                new_val = sim.execute_action(action, price)
                portfolio_values.append(new_val)

                # Compute risk-adjusted reward
                if len(portfolio_values) > 5:
                    recent_vals = np.array(portfolio_values[-5:])
                    vol = np.std(np.diff(recent_vals)) / (np.mean(recent_vals) + 1e-8)
                    dd = 1 - (recent_vals[-1] / np.max(recent_vals))
                else:
                    vol, dd = 0, 0

                raw_reward = np.log((new_val + 1e-8) / (prev_val + 1e-8))
                reward = alpha * raw_reward - beta * vol - gamma * dd
                ema_reward = ema_alpha * reward + (1 - ema_alpha) * ema_reward
                reward_norm = np.tanh(ema_reward * tanh_scale)
                daily_rewards.append(reward_norm)

                # Append to buffer
                states_buffer.append(state_tensor)
                actions_buffer.append(action)
                rewards_buffer.append(reward_norm)

                # Update model every reward_interval steps
                if len(rewards_buffer) >= reward_interval:
                    for s, a, r in zip(states_buffer, actions_buffer, rewards_buffer):
                        optimizer.zero_grad()
                        with autocast(device_type="cuda"):
                            pred = model(s)
                            target = pred.clone().detach()
                            target[0, a] = r
                            loss = loss_fn(pred, target)
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    # Clear buffers
                    states_buffer.clear()
                    actions_buffer.clear()
                    rewards_buffer.clear()

            # End-of-stock refinement
            final_val = sim.get_current_value(df.iloc[-1]["Close"])
            reward_final = np.log((final_val + 1e-8) / (sim.initial_cash + 1e-8))
            num_trades = np.sum(np.array(sim.actions) != 1)
            total_fitness += reward_final
            total_trades += num_trades
            num_stocks += 1

            print(f"  {stock_name}: value={final_val:.2f}, reward={reward_final*100:+.2f}%, trades={num_trades}")

            final_state = torch.tensor(sim.get_state(df.iloc[-1]), dtype=torch.float32, device=device).unsqueeze(0)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                pred = model(final_state)
                target = pred.clone().detach()
                target[0, torch.argmax(pred).item()] = reward_final
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if reward_final > best_fitness:
                best_fitness = reward_final
                best_model = model.state_dict().copy()

        if num_stocks > 0:
            avg_fit = (total_fitness / num_stocks) * 100
            avg_trades = total_trades / max(num_stocks, 1)
            print(f"📊 Epoch {epoch + 1}: Avg Fitness={avg_fit:+.2f}%, Avg Trades={avg_trades:.1f}, Best={best_fitness*100:+.2f}%")

    if best_model:
        model.load_state_dict(best_model)
        print(f"\n✅ Phase 2 complete. Best fitness {best_fitness*100:+.2f}% loaded.")

    return model

# ================= Phase Selection Wrapper =================
def train(model, data_files, phase="auto"):
    if phase.lower() == "phase0":
        return train_model_phase0(model, data_files)
    elif phase.lower() == "phase1":
        return train_model_phase1(model, data_files)
    elif phase.lower() == "phase2":
        return train_model_phase2(model, data_files, is_static_data=True)
    elif phase.lower() == "auto":
        model = train_model_phase1(model, data_files)
        model = train_model_phase2(model, data_files, is_static_data=True)
        return model
    else:
        raise ValueError("Invalid phase selection. Choose 'phase0', 'phase1', 'phase2', or 'auto'.")
