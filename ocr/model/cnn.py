"""
CNN architecture for OCR on EMNIST digits/alphanumeric.

Architecture overview:
  Input: 1x28x28 grayscale image
  Block 1: Conv(1→32, 3x3) → BN → ReLU → Conv(32→32, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.25)
  Block 2: Conv(32→64, 3x3) → BN → ReLU → Conv(64→64, 3x3) → BN → ReLU → MaxPool(2x2) → Dropout(0.25)
  Classifier: Flatten → FC(1024) → BN → ReLU → Dropout(0.5) → FC(62)
"""

import torch
import torch.nn as nn

EMNIST_CLASSES = (
    [str(i) for i in range(10)] +          # 0–9  → indices 0-9
    [chr(i) for i in range(65, 91)] +      # A–Z  → indices 10-35
    [chr(i) for i in range(97, 123)]       # a–z  → indices 36-61
)
 
def decode_prediction(index: int) -> str:
# convert model output index to corresponding character
    return EMNIST_CLASSES[index]
 
def decode_sequence(indices) -> str:
    # If indices is a tensor, convert to list first
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    return ''.join(EMNIST_CLASSES[i] for i in indices)

class OCRCNN(nn.Module):
    def __init__(self, num_classes: int = 62):
        # 62 for EMNIST digits/alphanumeric (0-9 + A-Z + a-z).
        super(OCRCNN, self).__init__()

        # block 1:
        # Input: (B, 1, 28, 28)
        # Output after pool: (B, 32, 12, 12)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),  # (B,32,26,26)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0),  # (B,32,24,24)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                 # (B,32,12,12)
            nn.Dropout2d(p=0.25),
        )

        # block 2:
        # Input: (B, 32, 12, 12)
        # Output after pool: (B, 64, 4, 4)  [12→10→8→4 after 2 conv + pool]
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),  # → (B,64,10,10)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),  # → (B,64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # → (B,64,4,4)
            nn.Dropout2d(p=0.25),
        )

        # connected classifier
        # size -> 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    # returns the number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    pass