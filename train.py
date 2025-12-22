import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleDNN
from datetime import datetime
import os

def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

    model = SimpleDNN()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss {loss.item():.4f}")

    # Deployment Step: Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model_mnist_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train()