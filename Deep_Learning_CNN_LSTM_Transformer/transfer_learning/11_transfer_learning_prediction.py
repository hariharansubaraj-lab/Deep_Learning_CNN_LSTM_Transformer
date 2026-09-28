import torch
import torch.nn as nn

from torchvision import datasets, transforms, models


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


# Dataset
test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform
)


# Model
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)

model.load_state_dict(
    torch.load(
        "resnet18_mnist.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Select image
image, actual_label = test_dataset[0]

image_input = image.unsqueeze(0).to(device)


# Prediction
with torch.no_grad():

    output = model(image_input)

    predicted = torch.argmax(
        output,
        dim=1
    ).item()


print("Actual Label:", actual_label)
print("Predicted Label:", predicted)


if actual_label == predicted:
    print("Prediction: Correct!")
else:
    print("Prediction: Wrong!")