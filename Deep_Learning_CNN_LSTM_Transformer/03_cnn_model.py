import torch
import torch.nn as nn


# ==========================================
# 1. CNN Model
# ==========================================

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature Extraction
        self.features = nn.Sequential(

            # First Convolution Layer
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),


            # Second Convolution Layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )


        # Classification
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


    # ======================================
    # Forward Pass
    # ======================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 2. Create Model
# ==========================================

model = CNNModel()

print(model)


# ==========================================
# 3. Test Model with Dummy Input
# ==========================================

dummy_input = torch.randn(1, 1, 28, 28)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)