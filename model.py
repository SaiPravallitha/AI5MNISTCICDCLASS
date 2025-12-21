import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        # Layer 1: Convolutional
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Added: Max Pooling to reduce 28x28 to 14x14
        self.pool = nn.MaxPool2d(2, 2) 
        
        # Layer 2: Fully Connected
        # Input is now 16 channels * 14 * 14 pixels = 3136
        self.fc1 = nn.Linear(16 * 14 * 14, 16) # Reduced hidden neurons to 16
        
        # Layer 3: Output layer
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply pooling here
        x = x.view(-1, 16 * 14 * 14)        # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x