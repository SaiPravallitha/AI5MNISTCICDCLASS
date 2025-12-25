import torch
import pytest
from model import SimpleDNN
from torchvision import datasets, transforms

def test_parameter_count():
    model = SimpleDNN()
    params = sum(p.numel() for p in model.parameters())
    assert params < 25000, f"Too many parameters: {params}"

def test_accuracy_threshold():
    model = SimpleDNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=1000)
    
    data, target = next(iter(test_loader))
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    accuracy = pred.eq(target.view_as(pred)).sum().item() / 1000
    assert accuracy >= 0.95, f"Accuracy too low: {accuracy}"

# --- 3 UNIQUE ADDITIONAL TESTS ---

def test_output_range():
    """Test 1: Check if output scores are not NaN or Infinite"""
    model = SimpleDNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert not torch.isnan(output).any(), "Model produced NaN outputs"
    assert not torch.isinf(output).any(), "Model produced Infinite outputs"

def test_input_batch_invariance():
    """Test 2: Check if model can handle different batch sizes (1 and 10)"""
    model = SimpleDNN()
    for batch_size in [1, 10]:
        dummy_input = torch.randn(batch_size, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (batch_size, 10)

def test_weight_initialization():
    """Test 3: Check if weights are properly initialized (not all zeros)"""
    model = SimpleDNN()
    for name, param in model.named_parameters():

        if 'bn' in name and 'bias' in name:
            continue

        assert torch.count_nonzero(param) > 0, f"Layer {name} is uninitialized"