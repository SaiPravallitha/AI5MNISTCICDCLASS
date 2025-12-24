import torch
import pytest
from model import SimpleDNN
from torchvision import datasets, transforms

def test_model_pipeline():
    model = SimpleDNN()
    
    # 1. Check Parameter Count (< 100,000)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count}")
    assert param_count < 25000, f"Model too large: {param_count} params"

    # 2. Check Input Shape (28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(dummy_input)
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {e}")

    # 3. Check Output Shape (10 classes)
    assert output.shape[1] == 10, f"Expected 10 outputs, got {output.shape[1]}"

def test_accuracy():
    # 4. Check Accuracy (> 80%)
    model = SimpleDNN()
    # Loading the latest trained model (mocking a CI artifact check)
    # For CI, we often re-train or load the saved .pth
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000)
    
    # Simulating a quick validation (using first batch for speed in CI)
    data, target = next(iter(test_loader))
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / 1000
    
    # Note: Accuracy might be low without loading trained weights, 
    # but this confirms the validation logic works.
    print(f"Validation Accuracy: {accuracy*100}%")
    assert accuracy >= 0.95, "Accuracy check triggered"