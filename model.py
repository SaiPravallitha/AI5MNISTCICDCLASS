import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        # Layer 1: 1 -> 8 channels (3x3 kernel)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Layer 2: 8 -> 16 channels (3x3 kernel)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Max Pooling to reduce size from 28x28 to 7x7
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two pools (28 -> 14 -> 7), input is 16 * 7 * 7 = 784
        self.fc1 = nn.Linear(784, 16) 
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x))) # 14x14 -> 7x7
        x = x.view(-1, 784)                  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x