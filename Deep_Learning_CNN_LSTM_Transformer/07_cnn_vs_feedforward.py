import torch
import torch.nn as nn

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent / "data"


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
# 3. MNIST Dataset
# ==========================================

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# ==========================================
# 4. Feed-Forward Network
# ==========================================

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 10)
        )


    def forward(self, x):

        return self.network(x)


# ==========================================
# 5. CNN Model
# ==========================================

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
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

        return self.classifier(x)


# ==========================================
# 6. Evaluation Function
# ==========================================

def evaluate_model(model):

    model.eval()

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


    accuracy = 100 * correct / total

    return accuracy


# ==========================================
# 7. Load or Train Feed-Forward Model
# ==========================================

feedforward_path = BASE_DIR / "feedforward_mnist.pth"


def train_feedforward_model(path: Path):
    model = FeedForwardNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(3):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Feed-forward training epoch {epoch + 1}/3 "
            f"- loss: {loss.item():.4f}"
        )

    torch.save(model.state_dict(), path)
    return model


if feedforward_path.exists():
    feedforward_model = FeedForwardNet().to(device)
    feedforward_model.load_state_dict(
        torch.load(
            feedforward_path,
            map_location=device
        )
    )
else:
    print(
        "Feed-forward checkpoint not found. "
        "Training a new model for comparison..."
    )
    feedforward_model = train_feedforward_model(feedforward_path)


# ==========================================
# 8. Load CNN Model
# ==========================================

cnn_model_path = BASE_DIR / "cnn_mnist.pth"

if not cnn_model_path.exists():
    raise FileNotFoundError(
        f"CNN checkpoint not found at: {cnn_model_path}. "
        "Please train the CNN model first."
    )

cnn_model = CNNModel().to(device)

cnn_model.load_state_dict(
    torch.load(
        cnn_model_path,
        map_location=device
    )
)


# ==========================================
# 9. Evaluate Both Models
# ==========================================

feedforward_accuracy = evaluate_model(
    feedforward_model
)

cnn_accuracy = evaluate_model(
    cnn_model
)


# ==========================================
# 10. Display Results
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print(
    f"Feed-Forward Accuracy : "
    f"{feedforward_accuracy:.2f}%"
)

print(
    f"CNN Accuracy          : "
    f"{cnn_accuracy:.2f}%"
)


# ==========================================
# 11. Compare Models
# ==========================================

if cnn_accuracy > feedforward_accuracy:

    print("\nCNN performs better than Feed-Forward Network.")

elif cnn_accuracy < feedforward_accuracy:

    print(
        "\nFeed-Forward Network performs better "
        "than CNN."
    )

else:

    print("\nBoth models have the same accuracy.")