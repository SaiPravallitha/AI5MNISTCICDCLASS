import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleDNN

def train():
    # Image Augmentation: Subtle rotations and shifts
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transform), 
        batch_size=64, shuffle=True)

    model = SimpleDNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Higher LR for 1-epoch speed
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "model.pth")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train()