
import os
import numpy as np
import torch
from PIL import Image

import model

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = "snapshots_100epochs/Epoch99.pth"
INPUT_DIR = "data/test_data/DICM"
OUTPUT_DIR = "comparison_my_model"
IMAGE_IDS = ["26", "29", "63"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load my independently trained Zero-DCE model
net = model.enhance_net_nopool().to(DEVICE)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)
net.load_state_dict(state_dict)
net.eval()

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}\n")

# BT.709 luminance, calculated on the 0-255 RGB scale
def calculate_metrics(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    return float(luminance.mean()), float(luminance.std())

def enhance_image(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        _, enhanced, _ = net(tensor)

    result = (
        enhanced.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .cpu()
        .numpy()
    )

    return Image.fromarray(
        (result * 255).round().astype(np.uint8)
    )

print(f"{'Image':<8}{'Model':<24}{'Brightness':>12}{'Contrast':>12}")
print("-" * 56)

for image_id in IMAGE_IDS:
    image_path = os.path.join(INPUT_DIR, f"{image_id}.jpg")

    original = Image.open(image_path).convert("RGB")
    enhanced = enhance_image(original)

    enhanced.save(os.path.join(OUTPUT_DIR, f"{image_id}_enhanced.png"))

    for label, image in [
        ("Original", original),
        ("My Trained Model", enhanced)
    ]:
        brightness, contrast = calculate_metrics(image)

        print(
            f"{image_id:<8}"
            f"{label:<24}"
            f"{brightness:>12.3f}"
            f"{contrast:>12.3f}"
        )

    print()

print(f"Enhanced images saved to: {OUTPUT_DIR}/")
