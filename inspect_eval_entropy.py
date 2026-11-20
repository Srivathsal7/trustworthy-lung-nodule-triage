import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_large import LargeNoduleDataset
from model_large import NoduleResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoduleResNet().to(device)
model.load_state_dict(torch.load("outputs/best_model_large.pth", map_location=device, weights_only=True))
model.train()

test_ds = LargeNoduleDataset(split="test", include_indeterminate=False)
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

results = []
print("Evaluating test set exactly as evaluate_large.py does...")

with torch.no_grad():
    for i, (images, labels, masks, meta) in enumerate(loader):
        images = images.to(device)
        preds = [F.softmax(model(images), dim=1).cpu().numpy() for _ in range(30)]
        mean_p = np.mean(preds, axis=0)[0]
        
        # Calculate natural log entropy and log2 entropy
        entropy_nat = float(-np.sum(mean_p * np.log(mean_p + 1e-8)))
        entropy_base2 = float(-np.sum(mean_p * np.log2(mean_p + 1e-8)))
        
        results.append({
            "idx": i,
            "label": "Malignant" if labels.item() == 1 else "Benign",
            "prob_mal": round(float(mean_p[1]), 4),
            "prob_ben": round(float(mean_p[0]), 4),
            "entropy_nat": round(entropy_nat, 4),
            "entropy_base2": round(entropy_base2, 4)
        })

df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("MEAN ENTROPY SUMMARY ACROSS TEST SET:")
print(f"Mean Natural Log Entropy: {df['entropy_nat'].mean():.4f}")
print(f"Mean Base-2 Entropy:     {df['entropy_base2'].mean():.4f}")
print("=" * 75)

print("\nTOP 5 HIGH-CONFIDENCE BENIGN CASES (Lowest Entropy):")
print(df[df["label"] == "Benign"].sort_values("entropy_nat").head(5).to_string(index=False))

print("\nTOP 5 HIGH-CONFIDENCE MALIGNANT CASES (Lowest Entropy):")
print(df[df["label"] == "Malignant"].sort_values("entropy_nat").head(5).to_string(index=False))
