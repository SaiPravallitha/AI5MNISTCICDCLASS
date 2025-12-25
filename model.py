import torch
import torch.nn as nn
import torch.nn.functional as F

# Just define the class directly
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 7 * 7, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 20 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x