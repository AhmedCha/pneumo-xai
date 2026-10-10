from pathlib import Path

import torch
from torch import nn, optim

from src.dataset import (
    train_loader,
    val_loader,
    train_dataset,
)
from src.model import build_model, DEVICE


# ============================================================
# 1. CONFIGURATION
# ============================================================

EPOCHS = 5
LEARNING_RATE = 0.0001

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "vgg16_best.pth"


# ============================================================
# 2. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    # Keep the frozen feature extractor in evaluation mode.
    model.features.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# ============================================================
# 3. VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# ============================================================
# 4. MAIN TRAINING PROCESS
# ============================================================

def main():
    print("Starting VGG16 training...", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Training images: {len(train_dataset)}", flush=True)
    print(f"Validation images: {len(val_loader.dataset)}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(pretrained=True)

    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that are not frozen.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}",
            flush=True,
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f}",
            flush=True,
        )

        print(
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                MODEL_PATH,
            )

            print(
                f"Saved best checkpoint: {MODEL_PATH}",
                flush=True,
            )

    print("\nTraining finished.", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()