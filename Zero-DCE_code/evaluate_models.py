
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)
import model

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoints = {
    "Authors pretrained": "snapshots/Epoch99.pth",
  
}
#   "My 1-epoch model": "snapshots_training_test/Epoch0.pth"
image_ids = ["26", "29", "63"]

def load_model(path):
    net = model.enhance_net_nopool().to(device)
    net.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    net.eval()
    return net

models = {
    name: load_model(path)
    for name, path in checkpoints.items()
}

def measure(image):
    # Input RGB image, float values in [0, 1]
    gray = (
        0.2126 * image[:, :, 0] +
        0.7152 * image[:, :, 1] +
        0.0722 * image[:, :, 2]
    )
    return {
        "Brightness": float(gray.mean() * 255),
        "Contrast": float(gray.std() * 255)
    }

rows = []

for image_id in image_ids:
    image_path = f"data/test_data/DICM/{image_id}.jpg"
    original = np.asarray(
        Image.open(image_path).convert("RGB"),
        dtype=np.float32
    ) / 255.0

    original_metrics = measure(original)

    rows.append({
        "Image": image_id,
        "Model": "Original",
        **original_metrics,
        "PSNR": np.nan,
        "SSIM": np.nan,
        "MAE": np.nan
    })

    tensor = torch.from_numpy(original).permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    for name, net in models.items():
        with torch.no_grad():
            _, enhanced, _ = net(tensor)

        output = enhanced.squeeze(0).permute(
            1, 2, 0
        ).cpu().numpy()

        output = np.clip(output, 0, 1)

        metrics = measure(output)

        psnr = peak_signal_noise_ratio(
            original, output, data_range=1.0
        )

        ssim = structural_similarity(
            original, output,
            channel_axis=2,
            data_range=1.0
        )

        mae = np.mean(np.abs(original - output))

        rows.append({
            "Image": image_id,
            "Model": name,
            **metrics,
            "PSNR": psnr,
            "SSIM": ssim,
            "MAE": mae * 255
        })

df = pd.DataFrame(rows)

numeric_cols = [
    "Brightness", "Contrast", "PSNR", "SSIM", "MAE"
]
df[numeric_cols] = df[numeric_cols].round(3)

os.makedirs("comparison_results", exist_ok=True)
df.to_csv("comparison_results/all_metrics.csv", index=False)

print(df.to_string(index=False))
print("\nSaved: comparison_results/all_metrics.csv")
print("Note: PSNR, SSIM and MAE use the dark input as reference.")
print("They measure change, not enhancement quality.")
