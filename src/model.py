import torch
from torch import nn
from torchvision import models


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. BUILD THE VGG16 MODEL
# ============================================================

def build_model(pretrained=True):
    """
    Create a VGG16 model for binary chest X-ray classification.

    Classes:
        0 -> NORMAL
        1 -> PNEUMONIA
    """

    if pretrained:
        weights = models.VGG16_Weights.DEFAULT
    else:
        weights = None

    model = models.vgg16(weights=weights)

    # Freeze the convolutional feature extractor initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the original ImageNet classification layer.
    input_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(
        in_features=input_features,
        out_features=NUM_CLASSES,
    )

    return model.to(DEVICE)


# ============================================================
# 3. TEST MODEL INITIALIZATION
# ============================================================

def main():
    print("Initializing VGG16...", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    model = build_model(pretrained=True)

    print("\nModel initialized successfully.", flush=True)
    print(f"Architecture: {model.__class__.__name__}", flush=True)
    print(f"Number of output classes: {NUM_CLASSES}", flush=True)

    # Verify the expected input and output shapes.
    sample_images = torch.randn(2, 3, 224, 224).to(DEVICE)

    model.eval()

    with torch.no_grad():
        predictions = model(sample_images)

    print("Input shape:", sample_images.shape, flush=True)
    print("Output shape:", predictions.shape, flush=True)
    print("Device:", next(model.parameters()).device, flush=True)


if __name__ == "__main__":
    main()