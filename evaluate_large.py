from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_large import LargeNoduleDataset
from model_large import NoduleResNet

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_accs, bin_confs = [], []

    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > b_low) & (confidences <= b_high)
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            ece += np.abs(conf - acc) * prop
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((b_low + b_high) / 2.0)

    return ece, bin_accs, bin_boundaries

def run_mc_dropout(model, images, num_samples=30):
    model.train()  # Keeps MC dropout layers active
    stoch_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            stoch_probs.append(probs.cpu().numpy())

    stoch_probs = np.stack(stoch_probs, axis=1)
    mean_probs = np.mean(stoch_probs, axis=1)
    eps = 1e-8
    entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    return mean_probs, entropy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("outputs/best_model_large.pth")

    model = NoduleResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    test_ds = LargeNoduleDataset(split="test", include_indeterminate=False)
    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    images, labels, _, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.numpy()

    mc_probs, mc_entropy = run_mc_dropout(model, images, num_samples=30)
    preds = np.argmax(mc_probs, axis=1)

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, mc_probs[:, 1])
    ece, bin_accs, bin_boundaries = compute_ece(mc_probs, labels, n_bins=10)
    cm = confusion_matrix(labels, preds)

    print("=" * 65)
    print(f"LARGE DATASET TEST COHORT (N = {len(labels)})")
    print(f"  • Test Accuracy:              {acc*100:.2f}%")
    print(f"  • Balanced Accuracy:          {bal_acc*100:.2f}%")
    print(f"  • Test F1 Score:              {f1:.3f}")
    print(f"  • Test ROC-AUC:               {auc:.3f}")
    print(f"  • Expected Calibration Error: {ece*100:.2f}%")
    print(f"\nConfusion Matrix:\n{cm}")

    # Evaluate Indeterminate (ambiguous) nodules
    indet_ds = LargeNoduleDataset(split="test", include_indeterminate=True)
    indet_df = indet_ds.df[indet_ds.df["binary_label"] == -1]
    if not indet_df.empty:
        indet_loader = DataLoader(indet_ds, batch_size=len(indet_ds), shuffle=False)
        all_imgs, all_lbls, _, _ = next(iter(indet_loader))
        indet_mask = (all_lbls.numpy() == -1)
        _, indet_entropies = run_mc_dropout(model, all_imgs[indet_mask].to(device), num_samples=30)

        print("\n[Uncertainty & Clinical Triage Benchmark]")
        print(f"  • Mean Predictive Entropy (Unambiguous Nodules): {np.mean(mc_entropy):.4f}")
        print(f"  • Mean Predictive Entropy (Indeterminate Nodules): {np.mean(indet_entropies):.4f}")
        print(f"  • Ambiguity Contrast Ratio:                       {np.mean(indet_entropies)/np.mean(mc_entropy):.2f}x")

    print("=" * 65)

if __name__ == "__main__":
    main()