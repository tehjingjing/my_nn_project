import torch.nn as nn


class CustomCNN(nn.Module):
    """Three convolution blocks and a two-layer classifier for 128x128 RGB images."""
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        # Learn increasingly detailed image features while reducing spatial size:
        # 128x128 -> 64x64 -> 32x32 -> 16x16.
        self.features = nn.Sequential(
            # First block detects simple patterns such as edges and color changes.
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # Deeper blocks combine earlier patterns into more complex leaf features.
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Convert the feature maps into a vector and classify it into leaf classes.
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            # Dropout regularises the dense layer to help limit overfitting.
            nn.Dropout(dropout_rate), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Extract image features, then return one raw score (logit) per class.
        return self.classifier(self.features(x))
