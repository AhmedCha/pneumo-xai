from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

# We use CPU for now. The code can also use CUDA if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# 4. CHECK DATASET DIRECTORIES
# ============================================================

def check_dataset_directories():
    print("\nChecking dataset directories...", flush=True)

    directories = {
        "Training": TRAIN_DIR,
        "Validation": VAL_DIR,
        "Testing": TEST_DIR,
    }

    for name, directory in directories.items():
        print(f"{name} directory: {directory}", flush=True)

        if not directory.exists():
            raise FileNotFoundError(
                f"{name} directory does not exist: {directory}"
            )


# ============================================================
# 5. LOAD DATASETS
# ============================================================

def load_datasets():
    print("\nLoading image datasets...", flush=True)

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

    # All splits must use the same class-to-index mapping.
    expected_classes = train_dataset.class_to_idx

    if val_dataset.class_to_idx != expected_classes:
        raise ValueError(
            "Training and validation datasets have different class mappings."
        )

    if test_dataset.class_to_idx != expected_classes:
        raise ValueError(
            "Training and test datasets have different class mappings."
        )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# 6. CREATE DATA LOADERS
# ============================================================

def create_data_loaders(train_dataset, val_dataset, test_dataset):
    print("\nCreating data loaders...", flush=True)

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

    return train_loader, val_loader, test_loader


# ============================================================
# 7. DISPLAY DATASET INFORMATION
# ============================================================

def display_dataset_information(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
):
    print("\n" + "=" * 55, flush=True)
    print("PNEUMONIA CLASSIFICATION - DATASET REPORT", flush=True)
    print("=" * 55, flush=True)

    print(f"\nDevice available for training: {DEVICE}", flush=True)
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)

    print("\nClasses:", train_dataset.classes, flush=True)
    print("Class mapping:", train_dataset.class_to_idx, flush=True)

    print("\nNumber of images:", flush=True)
    print(f"Training:   {len(train_dataset)}", flush=True)
    print(f"Validation: {len(val_dataset)}", flush=True)
    print(f"Testing:    {len(test_dataset)}", flush=True)

    print("\nNumber of batches:", flush=True)
    print(f"Training:   {len(train_loader)}", flush=True)
    print(f"Validation: {len(val_loader)}", flush=True)
    print(f"Testing:    {len(test_loader)}", flush=True)

    print("\nLoading one training batch...", flush=True)

    images, labels = next(iter(train_loader))

    print("Batch loaded successfully.", flush=True)
    print("Image tensor shape:", images.shape, flush=True)
    print("Label tensor shape:", labels.shape, flush=True)
    print("Labels in this batch:", labels.tolist(), flush=True)

    print("\nDataset initialization completed successfully.", flush=True)
    print("=" * 55, flush=True)


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():
    print("Starting dataset initialization...", flush=True)

    check_dataset_directories()

    train_dataset, val_dataset, test_dataset = load_datasets()

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    display_dataset_information(
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == "__main__":
    main()