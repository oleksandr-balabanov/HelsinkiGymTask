import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

@dataclass
class SimpleNNConfig:
    input_dim:int = 10
    hid_dim:int = 10
    output_dim:int = 10

class SimpleNN(nn.Module):
    def __init__(self, config:SimpleNNConfig = SimpleNNConfig()):
        super(SimpleNN, self).__init__()

        self.config = config
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config.input_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits