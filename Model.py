# model.py
import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 256, 128], output_size=1):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.network(x)

    def evaluate_fitness(self, fitness_func, *args, **kwargs):
        """Evaluate model fitness using provided fitness function"""
        self.eval()
        with torch.no_grad():
            fitness = fitness_func(self, *args, **kwargs)
        return float(fitness.cpu()) if isinstance(fitness, torch.Tensor) else fitness

    def backprop(self, loss_fn, optimizer, inputs, targets):
        """
        Perform a single backpropagation step.

        Args:
            loss_fn: Loss function (e.g., nn.MSELoss()).
            optimizer: Optimizer (e.g., torch.optim.Adam(self.parameters(), lr=0.001)).
            inputs: Input data (NumPy array or torch.Tensor).
            targets: Ground-truth labels (NumPy array or torch.Tensor).
        """
        self.train()
        optimizer.zero_grad()

        # Convert NumPy inputs/targets to tensors
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Forward pass
        outputs = self.forward(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        return float(loss.item())

# Utility functions
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=get_device()))