from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LargeNoduleDataset(Dataset):
    def __init__(self, manifest_path="data/processed_large/large_manifest.csv", split="train", augment=False, include_indeterminate=False):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        # Exclude indeterminate nodules (-1) during standard training & validation
        if not include_indeterminate:
            self.df = self.df[self.df["binary_label"] != -1].reset_index(drop=True)
            
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load pre-extracted (3, 64, 64) 2.5D patch and consensus mask
        with np.load(row["patch_file"]) as data:
            image = data["image"].astype(np.float32)  # shape: (3, 64, 64)
            mask = data["mask"].astype(np.float32)    # shape: (64, 64)

        # Spatial data augmentations for training
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()  # Horizontal flip
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()  # Vertical flip
                mask = np.flip(mask, axis=0).copy()

        label = int(row["binary_label"])
        return torch.from_numpy(image), label, torch.from_numpy(mask), row.to_dict()

if __name__ == "__main__":
    train_ds = LargeNoduleDataset(split="train", augment=True)
    val_ds = LargeNoduleDataset(split="val", augment=False)
    test_ds = LargeNoduleDataset(split="test", augment=False)
    
    print("=" * 65)
    print("DATASET SPLIT SUMMARY:")
    print(f"  • Train Set (Benign/Malignant): {len(train_ds)} samples")
    print(f"  • Val Set   (Benign/Malignant): {len(val_ds)} samples")
    print(f"  • Test Set  (Benign/Malignant): {len(test_ds)} samples")
    sample_img, sample_lbl, sample_mask, _ = train_ds[0]
    print(f"\nTensor Verification:")
    print(f"  • Image Tensor Shape: {sample_img.shape} (dtype: {sample_img.dtype})")
    print(f"  • Mask Tensor Shape:  {sample_mask.shape} (dtype: {sample_mask.dtype})")
    print(f"  • Label Value:        {sample_lbl}")
    print("=" * 65)