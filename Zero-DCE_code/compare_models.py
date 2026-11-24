
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import model

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

checkpoints = {
    "Authors pretrained": "snapshots/Epoch99.pth",
    "My 1-epoch model": "snapshots_training_test/Epoch0.pth"
}

image_paths = [
    "data/test_data/DICM/18.jpg",
    "data/test_data/DICM/20.jpg",
    "data/test_data/DICM/29.jpg",
    "data/test_data/DICM/35.jpg",
    "data/test_data/DICM/63.jpg"
]

os.makedirs("comparison_results", exist_ok=True)

def load_model(checkpoint):
    net = model.enhance_net_nopool().to(device)
    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )
    net.load_state_dict(state)
    net.eval()
    return net

models = {
    name: load_model(path)
    for name, path in checkpoints.items()
}

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)

    tensor = torch.from_numpy(
        original.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    results = [original]
    titles = ["Original"]

    with torch.no_grad():
        for name, net in models.items():
            _, enhanced, _ = net(tensor)
            output = enhanced.squeeze(0).permute(
                1, 2, 0
            ).cpu().numpy()
            results.append(np.clip(output, 0, 1))
            titles.append(name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, result, title in zip(axes, results, titles):
        ax.imshow(result)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    filename = os.path.basename(image_path)
    output_path = os.path.join(
        "comparison_results",
        os.path.splitext(filename)[0] + "_comparison.png"
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)
