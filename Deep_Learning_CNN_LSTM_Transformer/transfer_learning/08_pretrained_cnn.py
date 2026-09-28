import torch
import torch.nn as nn
from torchvision import models


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load pretrained ResNet18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Replace final classifier
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)


model = model.to(device)

print(model)