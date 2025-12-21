import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        # Layer 1: Convolutional
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Layer 2: Fully Connected
        self.fc1 = nn.Linear(16 * 28 * 28, 64)
        # Layer 3: Output layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x