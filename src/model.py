# ==========================================
# CIFAR-10 IMAGE CLASSIFICATION
# 6-LAYER VGG-STYLE CNN MODEL
# ==========================================

import torch
import torch.nn as nn


# Define our 6-Layer Convolutional Neural Network
class CIFAR10_CNN(nn.Module):

    def __init__(self):
        super(CIFAR10_CNN, self).__init__()

        # ------------------------------------------
        # BLOCK 1: Extracts basic features (edges, colors, textures)
        # Input image: (3, 32, 32) -> Output after MaxPool: (64, 16, 16)
        # ------------------------------------------
        self.conv_block1 = nn.Sequential(
            # Conv Layer 1: Takes 3 RGB channels, outputs 64 feature maps
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),      # Normalizes values to stabilize & speed up training
            nn.ReLU(inplace=True),   # Activation function (keeps positive values)

            # Conv Layer 2: Refines the 64 feature maps
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Reduces image height & width by half (32x32 -> 16x16)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Randomly drops 20% feature maps during training to prevent overfitting
            nn.Dropout2d(0.2)
        )

        # ------------------------------------------
        # BLOCK 2: Extracts mid-level features (shapes, ears, wheels, wings)
        # Input: (64, 16, 16) -> Output after MaxPool: (128, 8, 8)
        # ------------------------------------------
        self.conv_block2 = nn.Sequential(
            # Conv Layer 3: Increases depth from 64 to 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Conv Layer 4: Refines the 128 feature maps
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Reduces size by half again (16x16 -> 8x8)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Drops 30% feature maps for stronger regularization
            nn.Dropout2d(0.3)
        )

        # ------------------------------------------
        # BLOCK 3: Extracts high-level object parts (faces, ship hull, animal body)
        # Input: (128, 8, 8) -> Output after MaxPool: (256, 4, 4)
        # ------------------------------------------
        self.conv_block3 = nn.Sequential(
            # Conv Layer 5: Increases depth from 128 to 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv Layer 6: Refines the 256 feature maps
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Final size reduction (8x8 -> 4x4)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Drops 40% feature maps
            nn.Dropout2d(0.4)
        )

        # ------------------------------------------
        # CLASSIFIER HEAD (Dense / Fully Connected Layers)
        # Takes flattened 256 * 4 * 4 = 4096 features and predicts 10 classes
        # ------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # Converts 3D tensor (256, 4, 4) into 1D vector (4096)
            nn.Linear(256 * 4 * 4, 512),    # Dense layer combining all features into 512 neurons
            nn.BatchNorm1d(512),            # Stabilizes dense layer activations
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                # Drops 50% neurons to prevent memorization
            nn.Linear(512, 10)              # Final output layer: 10 raw scores (one per CIFAR-10 class)
        )

    # Defines how data flows through the network
    def forward(self, x):
        x = self.conv_block1(x)   # Pass through Block 1
        x = self.conv_block2(x)   # Pass through Block 2
        x = self.conv_block3(x)   # Pass through Block 3
        x = self.classifier(x)    # Pass through Dense Classifier
        return x


# Quick sanity test when running model.py directly
if __name__ == "__main__":
    model = CIFAR10_CNN()
    sample = torch.randn(64, 3, 32, 32)  # Dummy batch of 64 RGB images (32x32)
    out = model(sample)
    print("6-Layer CNN Output Shape:", out.shape)  # Should be (64, 10)
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))