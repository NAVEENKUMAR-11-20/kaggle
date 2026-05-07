"""
Train ResNet-18 classifier on Chihuahua vs Muffin dataset using 3LC.

- 3LC Table loading (train + val). By default uses .latest() so Dashboard
  edits are picked up automatically. Optional: load by explicit table URLs
  (see OPTION 2 in code, commented out).
- ResNet-18 training with weighted sampling (exclude undefined).
- Per-sample metrics and embeddings collection.
- Best model saved to best_model.pth (overwritten each run).

Usage:
    python register_tables.py  # Run once
    python train.py

Outputs:
    best_model.pth  - Best checkpoint by validation accuracy (overwritten each run).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
IMAGE_SIZE = 224  # ResNet-18 standard input size
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
NUM_CLASSES = 2  # chihuahua, muffin (undefined excluded from training)
CLASS_NAMES = ["chihuahua", "muffin", "undefined"]
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5
# Competition rule: train from scratch. No pretrained weights allowed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"ResNet-18: random init (no pretrained weights — competition rules)")


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[OK] Random seed set to {seed}")


# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    """ResNet-18 for Chihuahua vs Muffin (fixed architecture). Train from scratch — no pretrained weights (competition rule)."""
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        # No pretrained weights: competition requires training from scratch.
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_features, num_classes),
        )
        # Better weight initialization for training from scratch
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


def train_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return train_transform(image), sample["label"]


def val_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return val_transform(image), sample["label"]


# ============================================================================
# METRICS
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()
    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


# ============================================================================
# TRAINING
# ============================================================================

# Default: output path for best model (overwritten each run)
BEST_MODEL_FILENAME = "best_model.pth"


def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent
    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )
    print(f"[OK] Registered data path: {base_path.absolute()}")

    # -------------------------------------------------------------------------
    # Load 3LC Tables
    # -------------------------------------------------------------------------
    # OPTION 1 (default): Load by name with .latest() — uses newest revision
    # and picks up any edits made in the 3LC Dashboard.
    # OPTION 2 (commented out): Load by URL for a specific table revision.
    # Get URLs from Dashboard: Tables tab → select table → copy URL.
    # -------------------------------------------------------------------------
    print("\nLoading 3LC tables...")

    # OPTION 1: Load by name (recommended — automatic latest revision)
    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val",
    ).latest()

    # OPTION 2: Load by URL (uncomment and set URLs to use a specific revision)
    # TRAIN_TABLE_URL = "paste_your_train_table_url_here"
    # VAL_TABLE_URL = "paste_your_val_table_url_here"
    # train_table = tlc.Table.from_url(TRAIN_TABLE_URL)
    # val_table = tlc.Table.from_url(VAL_TABLE_URL)

    print(f"  Train: {len(train_table)} samples")
    print(f"  Val:   {len(val_table)} samples")
    print(f"  Train table URL: {train_table.url}")
    print(f"  Val table URL:   {val_table.url}")
    class_names = list(train_table.get_simple_value_map("label").values())
    print(f"  Classes: {class_names}")

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)
    train_sampler = train_table.create_sampler(exclude_zero_weights=True)
    train_dataloader = DataLoader(
        train_table,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )
    val_dataloader = DataLoader(val_table, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Cosine annealing with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    run = tlc.init(
        project_name=PROJECT_NAME,
        description="Chihuahua vs Muffin - data-centric workflow",
    )
    metric_schemas = {
        "loss": tlc.Schema(description="Cross entropy loss", value=tlc.Float32Value()),
        "predicted": tlc.CategoricalLabelSchema(display_name="predicted label", classes=class_names),
        "accuracy": tlc.Schema(description="Per-sample accuracy", value=tlc.Float32Value()),
        "confidence": tlc.Schema(description="Prediction confidence", value=tlc.Float32Value()),
    }
    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=metric_schemas,
    )
    indices_and_modules = list(enumerate(model.resnet.named_modules()))
    resnet_fc_layer_index = next((i for i, (n, _) in indices_and_modules if n == "fc"), len(indices_and_modules) - 1)
    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(layers=[resnet_fc_layer_index])
    predictor = tlc.Predictor(model, layers=[resnet_fc_layer_index])

    best_val_accuracy = 0.0
    best_model_state = None
    patience = 20  # Early stopping patience
    no_improve_count = 0
    print("\n" + "=" * 60)
    print("  Starting Training")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"  Image size: {IMAGE_SIZE} | Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Early stopping patience: {patience}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping for stable training from scratch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        avg_loss = running_loss / max(len(train_dataloader), 1)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = 100 * val_correct / val_total
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"  --> New best model! ({best_val_accuracy:.2f}%)")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        tlc.log({"epoch": epoch, "val_accuracy": val_accuracy, "train_loss": avg_loss, "train_accuracy": train_accuracy})

    print("\n" + "=" * 60)
    print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = base_path / BEST_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Best model saved to {model_path} (overwrites previous run)")

    print("\nCollecting metrics on train set...")
    model.eval()
    tlc.collect_metrics(
        train_table,
        predictor=predictor,
        metrics_collectors=[classification_metrics_collector, embeddings_metrics_collector],
        split="train",
        dataloader_args={"batch_size": BATCH_SIZE, "num_workers": 0},
    )
    print("\nReducing embeddings...")
    try:
        # Use UMAP (default); PaCMAP can fail with "_var_var_13" on some setups (e.g. PyTorch nightly / RTX 50).
        run.reduce_embeddings_by_foreign_table_url(
            train_table.url,
            method="umap",
            n_neighbors=15,
            n_components=3,
        )
        print("  [OK] Embeddings reduced (UMAP, 3D).")
    except Exception as e:
        print(f"  WARNING: Embedding reduction failed: {e}")
        print("  Training and metrics are saved. Run and table are still valid; only the embedding view may be missing in the Dashboard.")
    run.set_status_completed()
    print("\n[OK] Done. View results at 3LC Dashboard (run: 3lc service)")


if __name__ == "__main__":
    train()
