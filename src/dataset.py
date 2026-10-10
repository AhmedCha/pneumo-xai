from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_datasets():
    for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=image_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=image_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=image_transform,
    )

    expected_classes = train_dataset.class_to_idx

    if val_dataset.class_to_idx != expected_classes:
        raise ValueError(
            "Validation classes do not match training classes."
        )

    if test_dataset.class_to_idx != expected_classes:
        raise ValueError(
            "Test classes do not match training classes."
        )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# 4. INITIALIZE DATASETS AND LOADERS
# ============================================================

train_dataset, val_dataset, test_dataset = load_datasets()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# 5. TEST THE DATASET MODULE
# ============================================================

def main():
    print("Dataset initialization successful.")
    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)


if __name__ == "__main__":
    main()