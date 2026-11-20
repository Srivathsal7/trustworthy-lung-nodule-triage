from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_large import NoduleResNet

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

model = NoduleResNet().to(device)
model.load_state_dict(
    torch.load("outputs/best_model_large.pth", map_location=device, weights_only=True)
)
model.train()  # Keep MC Dropout active

# Load test split
manifest_path = Path("data/processed_large/large_manifest.csv")
df = pd.read_csv(manifest_path)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Scanning all {len(test_df)} test split nodules...")

records = []
with torch.no_grad():
    for idx, row in test_df.iterrows():
        # Load native [0, 1] patch
        with np.load(row["patch_file"]) as data:
            img = data["image"].astype(np.float32)

        tensor_in = torch.from_numpy(img).unsqueeze(0).to(device)

        # 30-pass Monte Carlo Dropout
        stoch_preds = [
            F.softmax(model(tensor_in), dim=1).cpu().numpy()
            for _ in range(30)
        ]

        mean_p = np.mean(stoch_preds, axis=0)[0]
        benign_prob = float(mean_p[0])
        malignant_prob = float(mean_p[1])
        entropy = float(-np.sum(mean_p * np.log(mean_p + 1e-8)))
        pred_label = "Malignant" if malignant_prob > benign_prob else "Benign"
        confidence = float(np.max(mean_p))

        records.append({
            "test_index": idx,
            "subject_id": row["subject_id"],
            "nodule_id": row["nodule_id"],
            "dropdown_display": f"{row['subject_id']} - {row['nodule_id']} ({row['label_name']})",
            "ground_truth": row["label_name"],
            "predicted_label": pred_label,
            "confidence_pct": round(confidence * 100, 2),
            "malignancy_prob": round(malignant_prob, 4),
            "predictive_entropy": round(entropy, 4)
        })

res_df = pd.DataFrame(records)

# Extract top 10 benign and top 10 malignant cases with lowest entropy
top_benign = res_df[res_df["ground_truth"] == "Benign"].sort_values("predictive_entropy").head(10)
top_malignant = res_df[res_df["ground_truth"] == "Malignant"].sort_values("predictive_entropy").head(10)

combined_top20 = pd.concat([top_benign, top_malignant]).reset_index(drop=True)

# Ensure outputs directory exists and save
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
csv_out_path = out_dir / "low_entropy_presentation_cases.csv"
combined_top20.to_csv(csv_out_path, index=False)

print("\n" + "=" * 95)
print(f"SUCCESS: Saved Top 20 Low-Entropy Cases to: {csv_out_path}")
print("=" * 95)
print(f"{'Subject & Nodule ID':<26} | {'Truth':<10} | {'Pred':<10} | {'Confidence':<12} | {'Entropy'}")
print("-" * 95)
for _, r in combined_top20.iterrows():
    print(f"{r['subject_id'] + ' ' + r['nodule_id']:<26} | {r['ground_truth']:<10} | {r['predicted_label']:<10} | {r['confidence_pct']:>6.1f}%     | {r['predictive_entropy']:.4f}")
print("=" * 95)