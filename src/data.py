# src/data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_scifar10_dataloaders(data_dir: str = '../data/', batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    Prepares the sequential grayscale CIFAR-10 dataset.
    Applies the following transformations:
    1. Grayscale: channel dim reduced 3 -> 1
    2. ToTensor: converts to a torch tensor [0,1]
    3. Normalize: standardizes pixel values to mu=0.5, sigma=0.5
    4. Lambda: flattends the (1,32,32) tensor to a (1024,1) sequence.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(1024,1))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader