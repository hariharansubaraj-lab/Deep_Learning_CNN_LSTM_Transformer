import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. Image Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==========================================
# 2. Load Training Dataset
# ==========================================

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


# ==========================================
# 3. Load Testing Dataset
# ==========================================

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# 4. Create DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# 5. Check One Batch
# ==========================================

images, labels = next(iter(train_loader))


print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First image shape:", images[0].shape)
print("First label:", labels[0].item())