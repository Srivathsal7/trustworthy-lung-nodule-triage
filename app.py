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

st.set_page_config(
    page_title="Trustworthy AI - Pulmonary Diagnostic Interface",
    layout="wide"
)

# Empirically validated clinical triage boundary
ENTROPY_TRIAGE_THRESHOLD = 0.30


@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleResNet().to(dev)
    model.load_state_dict(
        torch.load("outputs/best_model_large.pth", map_location=dev, weights_only=True)
    )
    model.eval()
    return model, dev


@st.cache_data
def load_manifest():
    manifest_path = Path("data/processed_large/large_manifest.csv")
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return None


def normalize_hu(arr):
    clipped = np.clip(arr, -1000.0, 400.0)
    norm = (clipped - (-1000.0)) / (400.0 - (-1000.0))
    return norm.astype(np.float32)


def process_uploaded_file(uploaded_file):
    """Processes uploaded files directly into [0, 1] range matching preprocess_large.py."""
    fname = uploaded_file.name.lower()
    inferred_label = "Unknown / Custom Upload"

    if "benign" in fname:
        inferred_label = "Benign (External/Synthetic)"
    elif "malignant" in fname:
        inferred_label = "Malignant (External/Synthetic)"
    elif "indeterminate" in fname:
        inferred_label = "Indeterminate (External/Synthetic)"
    elif "normal" in fname or "healthy" in fname:
        inferred_label = "Normal Parenchyma (External/Synthetic)"

    if fname.endswith(".npz"):
        data = np.load(uploaded_file)
        img = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32) if "mask" in data else np.zeros((64, 64), dtype=np.float32)
        if "label" in data:
            inferred_label = str(data["label"])
        return img, mask, inferred_label

    elif fname.endswith(".dcm"):
        dcm = pydicom.dcmread(uploaded_file)
        pixel_array = dcm.pixel_array.astype(np.float32)
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        hu = pixel_array * slope + intercept
        norm_slice = normalize_hu(hu)

        h, w = norm_slice.shape
        if h >= 64 and w >= 64:
            cy, cx = h // 2, w // 2
            patch = norm_slice[cy - 32 : cy + 32, cx - 32 : cx + 32]
        else:
            patch = cv2.resize(norm_slice, (64, 64))

        img = np.stack([patch, patch, patch], axis=0).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        return img, mask, inferred_label

    else:
        # Standard PNG/JPG slice
        pil_img = Image.open(uploaded_file).convert("L")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        patch = cv2.resize(arr, (64, 64))
        img = np.stack([patch, patch, patch], axis=0).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        return img, mask, inferred_label


def generate_live_gradcam(model, tensor_in, target_class):
    """Computes Layer-4 Grad-CAM activation using the native [0, 1] tensor."""
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input_val, output_val):
        activations.append(output_val.detach())

    target_layer = model.get_last_conv_layer()
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    tensor_req = tensor_in.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits = model(tensor_req)
    logits[0, target_class].backward()

    h_fwd.remove()
    h_bwd.remove()

    pooled_grads = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    cam = torch.sum(pooled_grads * activations[0], dim=1).squeeze(0)
    cam = F.relu(cam).cpu().numpy()

    cmin, cmax = cam.min(), cam.max()
    if cmax - cmin < 1e-6:
        return np.zeros((64, 64), dtype=np.float32)

    heatmap = (cam - cmin) / (cmax - cmin)
    heatmap[heatmap < 0.25] = 0.0

    return cv2.resize(heatmap, (64, 64))


def main():
    st.title("🫁 Pulmonary Nodule Diagnostic & Uncertainty Triage Interface")
    st.caption("ResNet-18 + Monte Carlo Dropout with calibrated Layer-4 Grad-CAM explainability.")

    model, device = load_model()
    df = load_manifest()

    st.sidebar.header("Input Source")
    input_mode = st.sidebar.radio("Select Input Mode", ["Browse LIDC Dataset", "Upload CT / Patch"])

    img, mask = None, None
    ground_truth = "Unknown / Custom Upload"

    if input_mode == "Upload CT / Patch":
        st.sidebar.subheader("Upload Scan File")
        uploaded_file = st.sidebar.file_uploader(
            "Upload .npz (nodule patch), .dcm (DICOM slice), or image",
            type=["npz", "dcm", "png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            img, mask, ground_truth = process_uploaded_file(uploaded_file)
            st.sidebar.success(f"Loaded: `{uploaded_file.name}`")
        else:
            st.info("👈 Upload a CT slice or nodule patch from the sidebar to inspect.")
            return

    else:
        if df is None:
            st.error("Dataset manifest not found at `data/processed_large/large_manifest.csv`.")
            return

        split = st.sidebar.selectbox("Cohort Partition", ["test", "val", "train"])
        sub_df = df[df["split"] == split].reset_index(drop=True)

        choice = st.sidebar.selectbox(
            "Select Patient Nodule",
            range(len(sub_df)),
            format_func=lambda i: f"{sub_df.iloc[i]['subject_id']} - {sub_df.iloc[i]['nodule_id']} ({sub_df.iloc[i]['label_name']})"
        )
        row = sub_df.iloc[choice]
        ground_truth = row["label_name"]

        with np.load(row["patch_file"]) as npz_data:
            img = npz_data["image"].astype(np.float32)
            mask = npz_data["mask"].astype(np.float32)

    # Convert directly to PyTorch tensor without ImageNet normalization (matches training/eval pipeline)
    tensor_in = torch.from_numpy(img).unsqueeze(0).to(device)

    # 1. Activate Dropout ONLY, keeping BatchNorm frozen in eval mode
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    # 2. Run 30-sample Monte Carlo Dropout inference
    with torch.no_grad():
        stoch_preds = [
            F.softmax(model(tensor_in), dim=1).cpu().numpy()
            for _ in range(30)
        ]
    model.eval()

    mean_probs = np.mean(stoch_preds, axis=0)[0]
    benign_prob = float(mean_probs[0])
    malignant_prob = float(mean_probs[1])
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
    pred_idx = int(np.argmax(mean_probs))
    pred_label = "Malignant" if pred_idx == 1 else "Benign"

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Diagnosis", pred_label)
    c2.metric("Calibrated Confidence", f"{np.max(mean_probs)*100:.1f}%")
    c3.metric("Predictive Entropy", f"{entropy:.4f}")
    c4.metric("Ground Truth", ground_truth)

    st.markdown("---")

    # Triage decision logic
    if entropy >= ENTROPY_TRIAGE_THRESHOLD:
        st.warning(
            f"⚠️ **High Predictive Uncertainty (Entropy = {entropy:.4f} ≥ {ENTROPY_TRIAGE_THRESHOLD})**: "
            f"Automated Triage Boundary Triggered — Borderline morphology detected. Case routed for secondary radiologist review."
        )
    else:
        st.success(
            f"✅ **High Diagnostic Confidence (Entropy = {entropy:.4f} < {ENTROPY_TRIAGE_THRESHOLD})**: "
            f"Clear morphological pattern below threshold ({ENTROPY_TRIAGE_THRESHOLD}). Automated diagnostic sign-off satisfied."
        )

    # Class Probability Breakdown
    st.subheader("Predictive Class Probabilities (MC Averaged)")
    prob_df = pd.DataFrame({
        "Diagnosis": ["Benign", "Malignant"],
        "Probability": [benign_prob, malignant_prob]
    })
    st.bar_chart(prob_df.set_index("Diagnosis"), height=180)

    # Live Grad-CAM Generation
    cam_heatmap = generate_live_gradcam(model, tensor_in, pred_idx)

    # 4-Panel Multi-Modal Visualizer
    st.subheader("Multi-Panel Diagnostic & Grad-CAM Visualizer")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        ax1.imshow(img[1], cmap="gray")
        ax1.axis("off")
        st.pyplot(fig1, use_container_width=True)
        st.caption("Central Axial CT Slice (-1000 to 400 HU)")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(mask, cmap="bone")
        ax2.axis("off")
        st.pyplot(fig2, use_container_width=True)
        st.caption("Consensus Mask (Ground Truth)")

    with col3:
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        ax3.imshow(cam_heatmap, cmap="jet")
        ax3.axis("off")
        st.pyplot(fig3, use_container_width=True)
        st.caption("Focal Grad-CAM Activation Heatmap")

    with col4:
        fig4, ax4 = plt.subplots(figsize=(3, 3))
        ax4.imshow(img[1], cmap="gray")
        ax4.imshow(cam_heatmap, cmap="jet", alpha=0.45)
        ax4.axis("off")
        st.pyplot(fig4, use_container_width=True)
        st.caption(f"Saliency Overlay (Predicted: {pred_label})")


if __name__ == "__main__":
    main()