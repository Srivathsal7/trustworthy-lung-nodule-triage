from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_large import NoduleResNet

# Page Configuration
st.set_page_config(
    page_title="Trustworthy AI - Pulmonary Diagnostic Interface",
    page_icon="🫁",
    layout="wide"
)

# Resolve paths relative to the repository root (works on Windows & Linux Cloud)
BASE_DIR = Path(__file__).resolve().parent
ENTROPY_TRIAGE_THRESHOLD = 0.30


def resolve_repo_path(raw_path_str: str) -> Path:
    """
    Converts Windows paths (e.g., C:\\Users\\...\\data\\processed_large\\file.npz
    or data\\processed_large\\file.npz) into OS-agnostic paths relative to BASE_DIR.
    """
    cleaned = str(raw_path_str).replace("\\", "/")
    if "data/" in cleaned:
        cleaned = "data/" + cleaned.split("data/", 1)[1]
    elif "outputs/" in cleaned:
        cleaned = "outputs/" + cleaned.split("outputs/", 1)[1]
    return BASE_DIR / cleaned


@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleResNet().to(dev)
    weights_path = BASE_DIR / "outputs" / "best_model_large.pth"
    state_dict = torch.load(weights_path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dev


@st.cache_data
def load_manifest():
    manifest_path = BASE_DIR / "data" / "processed_large" / "large_manifest.csv"
    if not manifest_path.exists():
        return None
    df = pd.read_csv(manifest_path)
    if "patch_file" in df.columns:
        df["resolved_patch"] = df["patch_file"].apply(lambda p: str(resolve_repo_path(p)))
        df["file_exists"] = df["resolved_patch"].apply(lambda p: Path(p).exists())
    return df


def normalize_hu(arr: np.ndarray) -> np.ndarray:
    """Clips raw CT Hounsfield Units to [-1000, 400] HU and scales to [0.0, 1.0]."""
    clipped = np.clip(arr, -1000.0, 400.0)
    return ((clipped - (-1000.0)) / (400.0 - (-1000.0))).astype(np.float32)


def process_uploaded_file(uploaded_file):
    fname = uploaded_file.name.lower()
    inferred_label = "Unknown / Custom Upload"

    if fname.endswith(".npz"):
        data = np.load(uploaded_file)
        img = data["image"].astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        mask = data["mask"].astype(np.float32) if "mask" in data else np.zeros((64, 64), dtype=np.float32)
        if "label_name" in data:
            inferred_label = str(data["label_name"])
        elif "label" in data:
            inferred_label = str(data["label"])
        return img, mask, inferred_label

    elif fname.endswith(".dcm"):
        dcm = pydicom.dcmread(uploaded_file)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        hu = dcm.pixel_array.astype(np.float32) * slope + intercept
        norm_slice = normalize_hu(hu)
        h, w = norm_slice.shape
        if h >= 64 and w >= 64:
            patch = norm_slice[h // 2 - 32 : h // 2 + 32, w // 2 - 32 : w // 2 + 32]
        else:
            patch = cv2.resize(norm_slice, (64, 64))
        img = np.stack([patch, patch, patch], axis=0).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        return img, mask, inferred_label

    else:
        arr = np.array(Image.open(uploaded_file).convert("L")).astype(np.float32) / 255.0
        patch = cv2.resize(arr, (64, 64))
        img = np.stack([patch, patch, patch], axis=0).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        return img, mask, inferred_label


def generate_live_gradcam(model, tensor_in, target_class: int) -> np.ndarray:
    """Generates noise-filtered Layer-4 Grad-CAM saliency map."""
    gradients, activations = [], []
    target_layer = model.get_last_conv_layer()

    h_fwd = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    h_bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))

    tensor_req = tensor_in.clone().detach().requires_grad_(True)
    model.eval()
    model.zero_grad()
    logits = model(tensor_req)
    logits[0, target_class].backward()

    h_fwd.remove()
    h_bwd.remove()

    pooled_grads = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    cam = F.relu(torch.sum(pooled_grads * activations[0], dim=1).squeeze(0)).cpu().numpy()

    if cam.max() - cam.min() < 1e-6:
        return np.zeros((64, 64), dtype=np.float32)

    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap[heatmap < 0.25] = 0.0  # Suppress diffuse background activation noise
    return cv2.resize(heatmap, (64, 64))


def main():
    st.title("🫁 Pulmonary Nodule Diagnostic & Uncertainty Triage Interface")
    st.caption("ResNet-18 + Monte Carlo Dropout with calibrated Layer-4 Grad-CAM explainability.")

    model, device = load_model()
    df = load_manifest()

    # Sidebar Controls
    st.sidebar.header("Input Source")
    has_local_patches = df is not None and "file_exists" in df.columns and df["file_exists"].any()

    default_modes = ["Browse LIDC Dataset", "Upload CT / Patch"] if has_local_patches else ["Upload CT / Patch", "Browse LIDC Dataset"]
    input_mode = st.sidebar.radio("Select Input Mode", default_modes)

    if input_mode == "Upload CT / Patch":
        uploaded_file = st.sidebar.file_uploader(
            "Upload .npz, .dcm, or image",
            type=["npz", "dcm", "png", "jpg", "jpeg"]
        )
        if uploaded_file is None:
            st.info("👈 Upload a CT slice (`.dcm`), 2.5D patch (`.npz`), or grayscale image from the sidebar to run inference.")
            return
        img, mask, ground_truth = process_uploaded_file(uploaded_file)

    else:
        if df is None or not has_local_patches:
            st.warning(
                "No `.npz` nodule patches were found in `data/processed_large/` on the server. "
                "Please switch to **Upload CT / Patch** in the sidebar, or push your `.npz` files to GitHub."
            )
            return

        # Filter to rows whose .npz files exist on disk so every dropdown selection works
        avail_df = df[df["file_exists"]].reset_index(drop=True)
        available_splits = [s for s in ["test", "val", "train"] if s in avail_df["split"].unique()]
        if not available_splits:
            available_splits = avail_df["split"].unique().tolist()

        split = st.sidebar.selectbox("Cohort Partition", available_splits)
        sub_df = avail_df[avail_df["split"] == split].reset_index(drop=True)

        choice = st.sidebar.selectbox(
            "Select Patient Nodule",
            range(len(sub_df)),
            format_func=lambda i: f"{sub_df.iloc[i]['subject_id']} - {sub_df.iloc[i]['nodule_id']} ({sub_df.iloc[i]['label_name']})"
        )
        row = sub_df.iloc[choice]
        ground_truth = str(row["label_name"])

        with np.load(row["resolved_patch"]) as npz_data:
            img = npz_data["image"].astype(np.float32)
            mask = npz_data["mask"].astype(np.float32) if "mask" in npz_data else np.zeros((64, 64), dtype=np.float32)

    # 30-Pass Monte Carlo Dropout (BatchNorm frozen in eval(), Dropout active)
    tensor_in = torch.from_numpy(img).unsqueeze(0).to(device)
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    with torch.no_grad():
        stoch_preds = [F.softmax(model(tensor_in), dim=1).cpu().numpy() for _ in range(30)]
    model.eval()

    mean_probs = np.mean(stoch_preds, axis=0)[0]
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
    pred_idx = int(np.argmax(mean_probs))
    pred_label = "Malignant" if pred_idx == 1 else "Benign"
    confidence_pct = float(np.max(mean_probs) * 100.0)

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Diagnosis", pred_label)
    c2.metric("Calibrated Confidence", f"{confidence_pct:.1f}%")
    c3.metric("Predictive Entropy", f"{entropy:.4f}")
    c4.metric("Ground Truth", ground_truth)

    st.divider()

    # Automated Triage Gate Banner
    if entropy >= ENTROPY_TRIAGE_THRESHOLD:
        st.warning(
            f"⚠️ **High Predictive Uncertainty (Entropy = {entropy:.4f} ≥ {ENTROPY_TRIAGE_THRESHOLD})**: "
            "Ambiguous morphological signature detected. Automated sign-off withheld — routed for secondary radiologist review."
        )
    else:
        st.success(
            f"✅ **High Diagnostic Confidence (Entropy = {entropy:.4f} < {ENTROPY_TRIAGE_THRESHOLD})**: "
            f"Clear morphological pattern below threshold ({ENTROPY_TRIAGE_THRESHOLD}). Automated diagnostic sign-off satisfied."
        )

    # MC-Averaged Probability Bar Chart
    st.subheader("Predictive Class Probabilities (MC Averaged)")
    prob_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": [float(mean_probs[0]), float(mean_probs[1])]
    }).set_index("Class")
    st.bar_chart(prob_df, height=220)

    # 4-Panel Diagnostic & Grad-CAM Visualizer
    st.subheader("Multi-Panel Diagnostic & Grad-CAM Visualizer")
    heatmap = generate_live_gradcam(model, tensor_in, pred_idx)
    central_slice = img[1] if img.shape[0] >= 2 else img[0]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    axes[0].imshow(central_slice, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Central Axial CT Slice (-1000 to 400 HU)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Consensus Mask (Ground Truth)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[2].set_title("Focal Grad-CAM Activation Heatmap", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(central_slice, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[3].set_title(f"Saliency Overlay (Predicted: {pred_label})", fontsize=9)
    axes[3].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
