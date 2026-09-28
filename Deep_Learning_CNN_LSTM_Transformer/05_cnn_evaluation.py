import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ==========================================
# 3. Test Dataset
# ==========================================

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# 4. CNN Model
# ==========================================

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),

            nn.ReLU(),

            nn.Linear(128, 10)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 5. Load Trained Model
# ==========================================

model = CNNModel().to(device)

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device
    )
)

model.eval()


# ==========================================
# 6. Evaluation
# ==========================================

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ==========================================
# 7. Test Accuracy
# ==========================================

accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)