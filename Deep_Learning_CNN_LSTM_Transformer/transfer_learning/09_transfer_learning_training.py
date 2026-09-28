import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


# ==========================================
# 3. Dataset
# ==========================================

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ==========================================
# 4. Pretrained ResNet18
# ==========================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Replace classifier
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)

model = model.to(device)


# ==========================================
# 5. Loss
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 6. Optimizer
# ==========================================

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# ==========================================
# 7. Training
# ==========================================

epochs = 1

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# ==========================================
# 8. Save
# ==========================================

torch.save(
    model.state_dict(),
    "resnet18_mnist.pth"
)

print("Transfer learning model saved!")