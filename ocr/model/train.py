"""
Usage:
    python train.py                        # defaults: 10 epochs, batch 128, lr 0.001
    python train.py --epochs 15 --lr 0.0005
    python train.py --noise gaussian       # train with Gaussian noise augmentation
    python train.py --noise snp            # train with salt-and-pepper noise augmentation
    python train.py --noise both           # train with noise augmentation

explanation:
  1. Downloads EMNIST automatically via torchvision
  2. Applies optional noise augmentation
  3. Trains the OCRCNN with Adam optimizer + cosine LR schedule
  4. Validates after every epoch and prints accuracy
  5. Saves the best model weights to `best_model.pth`
  6. Prints a final test-set evaluation
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import OCRCNN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# noise transformations:
class AddGaussianNoise:
    # Adds zero-mean Gaussian noise with specified stddev.
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


class AddSaltAndPepperNoise:
    # randomly sets pixels to 0 (pepper) or 1 (salt).
    def __init__(self, prob: float = 0.05):
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noisy = tensor.clone()
        # salt
        salt_mask = torch.rand_like(tensor) < (self.prob / 2)
        noisy[salt_mask] = 1.0
        # pepper
        pepper_mask = torch.rand_like(tensor) < (self.prob / 2)
        noisy[pepper_mask] = 0.0
        return noisy

    def __repr__(self):
        return f"AddSaltAndPepperNoise(prob={self.prob})"

# builds the appropriate transforms based on the noise mode
def build_transforms(noise_mode: str):
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    noise_transforms = []
    if noise_mode == "gaussian":
        noise_transforms = [AddGaussianNoise(std=0.1)]
    elif noise_mode == "snp":
        noise_transforms = [AddSaltAndPepperNoise(prob=0.05)]
    elif noise_mode == "both":
        noise_transforms = [
            AddGaussianNoise(std=0.08),
            AddSaltAndPepperNoise(prob=0.04),
        ]

    train_transform = transforms.Compose(base + noise_transforms)
    val_transform = transforms.Compose(base)

    return train_transform, val_transform

# training and evaluation loops
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train OCR CNN on MNIST")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--batch",     type=int,   default=128)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--noise",     type=str,   default="none",
                        choices=["none", "gaussian", "snp", "both"],
                        help="Noise augmentation mode for training data")
    parser.add_argument("--data_dir",  type=str,   default="./data",
                        help="Where to download / cache MNIST")
    parser.add_argument("--save_path", type=str,   default="best_model.pth",
                        help="Where to save the best model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise mode: {args.noise}\n")

    # --- Data ---
    train_tf, val_tf = build_transforms(args.noise)

    train_dataset = datasets.MNIST(args.data_dir, train=True,  download=True, transform=train_tf)
    val_dataset   = datasets.MNIST(args.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=0)

    # --- Model ---
    model = OCRCNN(num_classes=10).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # --- Optimizer + Schedule ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing decays LR smoothly to near-0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_acc = 0.0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        flag = " ← best" if val_acc > best_val_acc else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  {val_loss:>9.4f}  {val_acc:>7.2%}  {lr:>8.6f}{flag}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "noise_mode": args.noise,
            }, args.save_path)

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()