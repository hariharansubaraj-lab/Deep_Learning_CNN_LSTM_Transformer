import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Image Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ==========================================
# 3. Load MNIST Dataset
# ==========================================

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# 4. DataLoader
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
# 5. CNN Model
# ==========================================

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 6. Create Model
# ==========================================

model = CNNModel().to(device)

print(model)


# ==========================================
# 7. Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 8. Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 9. Training Configuration
# ==========================================

epochs = 5

train_losses = []
train_accuracies = []


# ==========================================
# 10. Training Loop
# ==========================================

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)


        # -------------------------------
        # Forward Pass
        # -------------------------------

        outputs = model(images)


        # -------------------------------
        # Calculate Loss
        # -------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # -------------------------------
        # Clear Gradients
        # -------------------------------

        optimizer.zero_grad()


        # -------------------------------
        # Backpropagation
        # -------------------------------

        loss.backward()


        # -------------------------------
        # Update Weights
        # -------------------------------

        optimizer.step()


        # -------------------------------
        # Calculate Accuracy
        # -------------------------------

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    # ======================================
    # Epoch Results
    # ======================================

    epoch_loss = (
        running_loss / len(train_loader)
    )

    epoch_accuracy = (
        100 * correct / total
    )


    train_losses.append(epoch_loss)

    train_accuracies.append(
        epoch_accuracy
    )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ==========================================
# 11. Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "cnn_mnist.pth"
)

print("\nModel saved successfully!")