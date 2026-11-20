# test_correct_mc.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_large import NoduleResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoduleResNet().to(device)
model.load_state_dict(torch.load("outputs/best_model_large.pth", map_location=device, weights_only=True))

# 1. CRITICAL: Put model in eval mode, but activate ONLY Dropout
model.eval()
for m in model.modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.train()

manifest_path = Path("data/processed_large/large_manifest.csv")
df = pd.read_csv(manifest_path)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Scanning test set with corrected BatchNorm/Dropout behavior...")

records = []
with torch.no_grad():
    for idx, row in test_df.iterrows():
        with np.load(row["patch_file"]) as data:
            img = data["image"].astype(np.float32)

        tensor_in = torch.from_numpy(img).unsqueeze(0).to(device)

        # 30-pass MC Dropout with frozen BatchNorm
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
            "subject_id": row["subject_id"],
            "nodule_id": row["nodule_id"],
            "ground_truth": row["label_name"],
            "pred_label": pred_label,
            "confidence": round(confidence * 100, 1),
            "mal_prob": round(malignant_prob, 3),
            "entropy": round(entropy, 4)
        })

res = pd.DataFrame(records)

print("\n" + "=" * 80)
print(f"OVERALL TEST ACCURACY: {(res['ground_truth'] == res['pred_label']).mean() * 100:.2f}%")
print(f"OVERALL MEAN PREDICTIVE ENTROPY: {res['entropy'].mean():.4f}")
print("=" * 80)

print("\nTOP 5 LOW-ENTROPY BENIGN CASES:")
print(res[res["ground_truth"] == "Benign"].sort_values("entropy").head(5).to_string(index=False))

print("\nTOP 5 LOW-ENTROPY MALIGNANT CASES:")
print(res[res["ground_truth"] == "Malignant"].sort_values("entropy").head(5).to_string(index=False))