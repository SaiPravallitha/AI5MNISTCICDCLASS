import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        # Layer 1: Conv -> BN -> Pool (Parameters: 80)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        # Layer 2: Conv -> BN -> Pool (Parameters: 1,168)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pools, 28x28 becomes 7x7. Flattened: 16 * 7 * 7 = 784
        # Layer 3: FC (Parameters: 12,560)
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 28 -> 14
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 14 -> 7
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x