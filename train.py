import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleDNN

def train():
    # Milder Augmentation: Focus on clean digits for fast learning
    train_transform = transforms.Compose([
        transforms.RandomRotation(5), # Reduced from 10
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transform), 
        batch_size=64, shuffle=True)

    model = SimpleDNN()
    # Lower Learning Rate: 0.001 is much more stable for MNIST
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx}: Loss {loss.item():.4f}")
    
    torch.save(model.state_dict(), "model.pth")
    print("Training Complete. Model saved as model.pth")

if __name__ == "__main__":
    train()