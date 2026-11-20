from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_large import LargeNoduleDataset
from model_large import NoduleResNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"TRAINING INITIALIZED ON: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    train_ds = LargeNoduleDataset(split="train", augment=True)
    val_ds = LargeNoduleDataset(split="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Class balance weights for cross entropy
    labels = train_ds.df["binary_label"].values
    c0 = np.sum(labels == 0)
    c1 = np.sum(labels == 1)
    w0 = len(labels) / (2.0 * max(c0, 1))
    w1 = len(labels) / (2.0 * max(c1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"Class distribution - Benign: {c0}, Malignant: {c1} | Loss Weights: [{w0:.2f}, {w1:.2f}]")

    model = NoduleResNet(num_classes=2, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda")

    best_auc = 0.0
    checkpoint_path = out_dir / "best_model_large.pth"

    for epoch in range(1, 31):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, targets, _, _ in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()
        train_acc = (correct / total) * 100.0

        # Validation phase
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    probs = torch.softmax(model(images), dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_targets.extend(targets.numpy())

        val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.5
        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        saved_flag = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), checkpoint_path)
            saved_flag = "★ [Best Checkpoint Saved]"

        print(f"Epoch [{epoch:02d}/30] | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% || Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} {saved_flag}")

    print("=" * 70)
    print(f"TRAINING COMPLETE. Optimal Validation AUC: {best_auc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()