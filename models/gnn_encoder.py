import torch
import torch.nn as nn


class GNNEncoder(nn.Module):

    def __init__(self, input_dim=75, hidden_dim=128):

        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        return x
