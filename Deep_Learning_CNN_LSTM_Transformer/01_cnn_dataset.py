import torch
from torchvision import datasets, transforms


# Check device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Download MNIST training dataset
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True
)


# Download MNIST testing dataset
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True
)


# Dataset information
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# Check one sample
image, label = train_dataset[0]

print("Image shape:", image.size)
print("Label:", label)