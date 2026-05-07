"""
Standalone training script (no 3LC dependency) for fast iteration.
Uses the cleaned data directly from folder structure.

Usage:
    uv run train_standalone.py

After achieving good results, run the full train.py with 3LC for final submission.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 2
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5
PATIENCE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    """ResNet-18 trained from scratch (competition rule: no pretrained weights)."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_features, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)


# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================================
# TTA (Test-Time Augmentation) for validation
# ============================================================================

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ============================================================================
# TRAINING
# ============================================================================

def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent

    # Use clean data if available, otherwise use original
    train_dir = base_path / "data" / "train_clean"
    val_dir = base_path / "data" / "val_clean"
    if not train_dir.exists():
        train_dir = base_path / "data" / "train"
    if not val_dir.exists():
        val_dir = base_path / "data" / "val"

    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")

    # Load datasets
    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Balanced sampling
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    best_val_accuracy = 0.0
    best_model_state = None
    no_improve_count = 0

    print(f"\n{'='*60}")
    print(f"  Training ResNet-18 from scratch")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"  Image: {IMAGE_SIZE}x{IMAGE_SIZE} | Smoothing: {LABEL_SMOOTHING}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {lr:.6f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"  --> Best model! ({best_val_accuracy:.2f}%)")
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*60}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = base_path / "best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved to {model_path}")

    return best_val_accuracy


if __name__ == "__main__":
    train()
