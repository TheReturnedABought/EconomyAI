import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Model import Model as m
from Model import get_device as gd
from Model import save_model as sm
from Model import load_model as lm


# Trading simulation class
class TradingSimulation:
    def __init__(self, initial_cash=100.0):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = [self.initial_cash]
        self.actions = []
        self.current_step = 0

    def execute_action(self, action, price):
        """
        Execute trading action
        0: Sell, 1: Hold, 2: Buy
        """
        if action == 2:  # Buy
            if self.cash > 0:
                shares_to_buy = self.cash / price
                self.shares += shares_to_buy
                self.cash = 0

        elif action == 0:  # Sell
            if self.shares > 0:
                self.cash += self.shares * price
                self.shares = 0

        # For hold (action == 1), do nothing

        # Calculate current portfolio value
        current_value = self.cash + (self.shares * price)
        self.portfolio_value.append(current_value)
        self.actions.append(action)
        self.current_step += 1

        return current_value

    def get_current_value(self, price):
        return self.cash + (self.shares * price)

    def get_state(self, current_data):
        """
        Prepare state for the model
        Includes: Close price, moving averages, difference, and portfolio info
        """
        state = [
            current_data['Close'],
            current_data['5DayMAV'] if not pd.isna(current_data['5DayMAV']) else current_data['Close'],
            current_data['20DayMAV'] if not pd.isna(current_data['20DayMAV']) else current_data['Close'],
            current_data['DIFof2Av'] if not pd.isna(current_data['DIFof2Av']) else 0,
            self.cash / self.initial_cash,  # Normalized cash
            self.shares * current_data['Close'] / self.initial_cash  # Normalized share value
        ]
        return np.array(state, dtype=np.float32)


# Buy and Hold strategy for comparison
class BuyAndHold:
    def __init__(self, initial_cash=100.0):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = [self.initial_cash]
        self.buy_executed = False

    def execute_strategy(self, price):
        """
        Execute buy and hold strategy
        Buy at first opportunity, then hold
        """
        if not self.buy_executed and self.cash > 0:
            # Buy all shares at first opportunity
            shares_to_buy = self.cash / price
            self.shares += shares_to_buy
            self.cash = 0
            self.buy_executed = True

        # Calculate current portfolio value
        current_value = self.cash + (self.shares * price)
        self.portfolio_value.append(current_value)

        return current_value

    def get_current_value(self, price):
        return self.cash + (self.shares * price)


# Percentage-based fitness function
def calculate_fitness_percentage(simulation, final_price, initial_cash=100.0):
    """
    Calculate fitness as percentage increase in value
    """
    final_value = simulation.get_current_value(final_price)
    percentage_increase = ((final_value - initial_cash) / initial_cash) * 100
    return percentage_increase


# Improved fitness function with risk adjustment
def calculate_improved_fitness(simulation, prices, initial_cash=100.0):
    """
    Improved fitness calculation with multiple factors
    Returns percentage increase with risk adjustment
    """
    final_value = simulation.get_current_value(prices[-1])

    # Calculate returns
    total_return_percentage = ((final_value - initial_cash) / initial_cash) * 100

    # Calculate volatility (risk)
    portfolio_values = simulation.portfolio_value
    returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
        returns.append(daily_return)

    if len(returns) > 1:
        volatility = np.std(returns) * 100  # as percentage
    else:
        volatility = 0

    # Calculate maximum drawdown
    peak = initial_cash
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Penalize excessive trading
    actions = simulation.actions
    trade_count = actions.count(0) + actions.count(2)  # Count buys and sells
    trade_penalty = (trade_count / len(actions)) * 10 if actions else 0  # Penalty up to 10%

    # Risk-adjusted fitness
    risk_adjusted_fitness = total_return_percentage - (volatility * 0.1) - (max_drawdown * 50) - trade_penalty

    return risk_adjusted_fitness, total_return_percentage


# Training function with per-CSV reporting
def train_model(model, data_files, num_epochs=100, learning_rate=0.001, use_improved_fitness=False):
    device = gd()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_fitness = -float('inf')
    best_model = None

    for epoch in range(num_epochs):
        epoch_total_fitness = 0
        epoch_total_percentage = 0
        num_simulations = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print("-" * 50)

        for file_path in data_files:
            df = pd.read_csv(file_path)
            stock_name = file_path.split('/')[-1].split('.')[0]

            # Skip files with insufficient data
            if len(df) < 30:
                continue

            simulation = TradingSimulation()

            # Prepare training data
            states = []
            targets = []

            for i in range(len(df)):
                current_data = df.iloc[i]

                # Skip rows with missing data
                if pd.isna(current_data['Close']) or pd.isna(current_data['5DayMAV']):
                    continue

                # Get current state
                state = simulation.get_state(current_data)
                states.append(state)

                # Model makes decision (0: sell, 1: hold, 2: buy)
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    prediction = model(state_tensor)
                    action = torch.argmax(prediction).item()

                # Execute action
                simulation.execute_action(action, current_data['Close'])

                # Create target based on optimal action (simplified heuristic)
                price_trend = 1 if current_data['DIFof2Av'] > 0 else 0 if pd.isna(current_data['DIFof2Av']) else -1
                optimal_action = 2 if price_trend > 0 else 0 if price_trend < 0 else 1

                target = torch.zeros(3, device=device)
                target[optimal_action] = 1.0
                targets.append(target)

            if states and targets:
                # Convert to tensors
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                targets_tensor = torch.stack(targets)

                # Train model
                loss = model.backprop(loss_fn, optimizer, states_tensor, targets_tensor)

                # Calculate fitness
                if use_improved_fitness:
                    fitness, percentage = calculate_improved_fitness(simulation, df['Close'].values)
                else:
                    percentage = calculate_fitness_percentage(simulation, df.iloc[-1]['Close'])
                    fitness = percentage  # Use simple percentage for fitness

                # Print individual stock performance
                print(f"  {stock_name}: {percentage:+.2f}%")

                epoch_total_fitness += fitness
                epoch_total_percentage += percentage
                num_simulations += 1

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_model = model.state_dict().copy()

        if num_simulations > 0:
            avg_fitness = epoch_total_fitness / num_simulations
            avg_percentage = epoch_total_percentage / num_simulations
            print(f"\n  Epoch Summary:")
            print(f"  Average Fitness: {avg_fitness:+.2f}%")
            print(f"  Average Return: {avg_percentage:+.2f}%")
            print(f"  Best Fitness So Far: {best_fitness:+.2f}%")

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"\nLoaded best model with fitness: {best_fitness:+.2f}%")

    return model


# Evaluation function with buy-and-hold comparison
def evaluate_model(model, data_files, use_improved_fitness=False):
    results = {}

    for file_path in data_files:
        df = pd.read_csv(file_path)
        stock_name = file_path.split('/')[-1].split('.')[0]

        # Model simulation
        simulation = TradingSimulation()
        decisions = []

        # Buy-and-hold simulation
        buy_hold = BuyAndHold()
        buy_hold_values = [buy_hold.initial_cash]

        for i in range(len(df)):
            current_data = df.iloc[i]

            if pd.isna(current_data['Close']) or pd.isna(current_data['5DayMAV']):
                continue

            # Model decision
            state = simulation.get_state(current_data)

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=gd()).unsqueeze(0)
                prediction = model(state_tensor)
                action = torch.argmax(prediction).item()

            simulation.execute_action(action, current_data['Close'])

            # Buy-and-hold execution
            buy_hold.execute_strategy(current_data['Close'])
            buy_hold_values.append(buy_hold.get_current_value(current_data['Close']))

            decisions.append({
                'date': current_data['Date'],
                'close': current_data['Close'],
                'action': ['SELL', 'HOLD', 'BUY'][action],
                'portfolio_value': simulation.get_current_value(current_data['Close']),
                'buy_hold_value': buy_hold.get_current_value(current_data['Close'])
            })

        final_value = simulation.get_current_value(df.iloc[-1]['Close'])
        final_buy_hold_value = buy_hold.get_current_value(df.iloc[-1]['Close'])

        # Calculate percentages
        model_return_percentage = ((final_value - 100) / 100) * 100
        buy_hold_return_percentage = ((final_buy_hold_value - 100) / 100) * 100
        outperformance = model_return_percentage - buy_hold_return_percentage

        results[stock_name] = {
            'decisions': decisions,
            'final_value': final_value,
            'buy_hold_final_value': final_buy_hold_value,
            'model_return_percentage': model_return_percentage,
            'buy_hold_return_percentage': buy_hold_return_percentage,
            'outperformance_percentage': outperformance
        }

    return results


# Function to run buy-and-hold only for quick comparison
def run_buy_and_hold_analysis(data_files):
    results = {}

    for file_path in data_files:
        df = pd.read_csv(file_path)
        stock_name = file_path.split('/')[-1].split('.')[0]

        buy_hold = BuyAndHold()

        for i in range(len(df)):
            current_data = df.iloc[i]

            if pd.isna(current_data['Close']):
                continue

            buy_hold.execute_strategy(current_data['Close'])

        final_value = buy_hold.get_current_value(df.iloc[-1]['Close'])
        return_percentage = ((final_value - 100) / 100) * 100

        results[stock_name] = {
            'final_value': final_value,
            'return_percentage': return_percentage
        }

    return results


# Main execution
if __name__ == "__main__":
    # Get all CSV files
    data_files = glob.glob('./DATA/*.csv')
    print(f"Found {len(data_files)} data files")

    if not data_files:
        print("No data files found!")
        exit()

    # Initialize model
    input_size = 6  # [Close, 5DayMAV, 20DayMAV, DIFof2Av, cash_ratio, shares_value_ratio]
    output_size = 3  # [Sell, Hold, Buy]

    model = m(input_size, output_size=output_size)

    # Quick buy-and-hold analysis on all files
    print("\nRunning Buy-and-Hold analysis on all stocks...")
    bh_results = run_buy_and_hold_analysis(data_files)

    total_bh_return = 0
    for stock, result in bh_results.items():
        total_bh_return += result['return_percentage']
        print(f"  {stock}: {result['return_percentage']:+.2f}%")

    avg_bh_return = total_bh_return / len(bh_results) if bh_results else 0
    print(f"Average Buy-and-Hold Return: {avg_bh_return:+.2f}%")

    # Train model with percentage-based fitness
    print("\nStarting training with percentage-based fitness...")
    trained_model = train_model(model, data_files, num_epochs=50, use_improved_fitness=False)

    # Save trained model
    sm(trained_model, 'trained_trading_model.pth')
    print("Model saved as 'trained_trading_model.pth'")

    # Evaluate model with buy-and-hold comparison
    print("\nEvaluating model with Buy-and-Hold comparison...")
    results = evaluate_model(trained_model, data_files[:5])  # Evaluate on first 5 files

    # Print results
    total_model_return = 0
    total_bh_comparison_return = 0
    total_outperformance = 0

    print("\n" + "=" * 60)
    print("DETAILED RESULTS:")
    print("=" * 60)

    for stock, result in results.items():
        print(f"\n{stock}:")
        print(f"  Model:     {result['model_return_percentage']:+.2f}%")
        print(f"  Buy&Hold:  {result['buy_hold_return_percentage']:+.2f}%")
        print(f"  Outperformance: {result['outperformance_percentage']:+.2f}%")

        total_model_return += result['model_return_percentage']
        total_bh_comparison_return += result['buy_hold_return_percentage']
        total_outperformance += result['outperformance_percentage']

    # Calculate averages
    avg_model_return = total_model_return / len(results) if results else 0
    avg_bh_comparison_return = total_bh_comparison_return / len(results) if results else 0
    avg_outperformance = total_outperformance / len(results) if results else 0

    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Average Model Return: {avg_model_return:+.2f}%")
    print(f"Average Buy-and-Hold Return: {avg_bh_comparison_return:+.2f}%")
    print(f"Average Outperformance: {avg_outperformance:+.2f}%")

    # Test with AIR.csv specifically
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS FOR AIR.CSV:")
    print("=" * 60)

    air_results = evaluate_model(trained_model, ['./DATA/AIR.csv'])
    for stock, result in air_results.items():
        print(f"\n{stock}:")
        print(f"  Model Return: {result['model_return_percentage']:+.2f}%")
        print(f"  Buy-and-Hold Return: {result['buy_hold_return_percentage']:+.2f}%")
        print(f"  Outperformance: {result['outperformance_percentage']:+.2f}%")

        # Count actions
        actions = [d['action'] for d in result['decisions']]
        buy_count = actions.count('BUY')
        sell_count = actions.count('SELL')
        hold_count = actions.count('HOLD')

        print(f"  Model Actions - BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")