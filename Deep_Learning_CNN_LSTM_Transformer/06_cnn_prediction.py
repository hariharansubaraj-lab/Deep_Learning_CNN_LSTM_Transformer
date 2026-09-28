import torch
import torch.nn as nn

from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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
# 3. Load Test Dataset
# ==========================================

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# 4. CNN Model
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
# 5. Create Model
# ==========================================

model = CNNModel().to(device)


# ==========================================
# 6. Load Trained Weights
# ==========================================

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device
    )
)


# Evaluation mode
model.eval()


# ==========================================
# 7. Select One Image
# ==========================================

image, actual_label = test_dataset[0]


# ==========================================
# 8. Add Batch Dimension
# ==========================================

image_input = image.unsqueeze(0).to(device)


print("Input shape:", image_input.shape)


# ==========================================
# 9. Prediction
# ==========================================

with torch.no_grad():

    output = model(image_input)

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()


# ==========================================
# 10. Display Result
# ==========================================

print("Actual Label:", actual_label)

print("Predicted Label:", predicted_class)


# ==========================================
# 11. Check Prediction
# ==========================================

if predicted_class == actual_label:

    print("Prediction: Correct!")

else:

    print("Prediction: Wrong!")


# ==========================================
# 12. Display Image
# ==========================================

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"Actual: {actual_label} | "
    f"Predicted: {predicted_class}"
)

plt.axis("off")

plt.show()