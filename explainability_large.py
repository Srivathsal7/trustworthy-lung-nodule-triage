from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_large import NoduleResNet

def generate_cam(model, tensor_in, target_class):
    gradients, activations = [], []

    def backward_hook(m, gi, go):
        gradients.append(go[0].detach())

    def forward_hook(m, i, o):
        activations.append(o.detach())

    target_layer = model.get_last_conv_layer()
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(tensor_in)
    logits[0, target_class].backward()

    h_fwd.remove()
    h_bwd.remove()

    pooled = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    cam = torch.sum(pooled * activations[0], dim=1).squeeze(0)
    cam = F.relu(cam).cpu().numpy()
    cmin, cmax = cam.min(), cam.max()
    heatmap = (cam - cmin) / (cmax - cmin + 1e-8)
    return cv2.resize(heatmap, (64, 64))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleResNet().to(device)
    model.load_state_dict(torch.load("outputs/best_model_large.pth", map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv("data/processed_large/large_manifest.csv")
    test_df = df[df["split"] == "test"]

    # Select representative samples
    samples = []
    for lbl in [0, 1, -1]:
        subset = test_df[test_df["binary_label"] == lbl]
        if not subset.empty:
            samples.append(subset.iloc[0])

    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 3.5 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, row in enumerate(samples):
        npz = np.load(row["patch_file"])
        img, mask = npz["image"], npz["mask"]
        t_in = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_class = int(torch.argmax(model(t_in), dim=1).item())
        heatmap = generate_cam(model, t_in, pred_class)

        axes[i, 0].imshow(img[1], cmap="gray")
        axes[i, 0].set_title(f"CT Slice ({row['label_name']})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="bone")
        axes[i, 1].set_title("Consensus Mask (4-Rad)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(heatmap, cmap="jet")
        axes[i, 2].set_title("Grad-CAM Activation")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img[1], cmap="gray")
        axes[i, 3].imshow(heatmap, cmap="jet", alpha=0.45)
        pred_txt = "Malignant" if pred_class == 1 else "Benign"
        axes[i, 3].set_title(f"Overlay (Pred: {pred_txt})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_file = Path("outputs/gradcam_large.png")
    plt.savefig(out_file, dpi=200)
    print("=" * 65)
    print(f"Saved explainability figure to: {out_file.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()