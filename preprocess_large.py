import configparser
# 1. Fix for Python 3.12+ configparser
configparser.SafeConfigParser = configparser.ConfigParser

import numpy as np
# 2. Fix for NumPy removing np.int and np.float aliases used inside pylidc
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

from pathlib import Path
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

def normalize_hu(arr):
    clipped = np.clip(arr, -1000.0, 400.0)
    norm = (clipped - (-1000.0)) / (400.0 - (-1000.0))
    return norm.astype(np.float32)

def main():
    out_dir = Path("data/processed_large")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("QUERYING SCANS IN LIDC-IDRI VIA PYLIDC...")
    print("=" * 65)

    scans = pl.query(pl.Scan).all()
    print(f"Total scans available in database: {len(scans)}")
    if len(scans) == 0:
        print("[ERROR] No scans detected. Ensure DICOM files exist in Downloads.")
        return

    records = []

    for scan in tqdm(scans, desc="Processing Scans"):
        patient_id = scan.patient_id
        try:
            nods = scan.cluster_annotations()
        except Exception:
            continue

        if not nods:
            continue

        try:
            vol = scan.to_volume(verbose=False)
        except Exception:
            continue

        for n_idx, nodule in enumerate(nods):
            mal_scores = [ann.malignancy for ann in nodule]
            mean_mal = float(np.mean(mal_scores))

            if mean_mal < 2.5:
                label_name = "Benign"
                binary_label = 0
            elif mean_mal > 3.5:
                label_name = "Malignant"
                binary_label = 1
            else:
                label_name = "Indeterminate"
                binary_label = -1

            try:
                center_coords = np.mean([ann.centroid for ann in nodule], axis=0).astype(int)
            except Exception:
                continue

            cy, cx, cz = center_coords[0], center_coords[1], center_coords[2]

            # Bounds check for 64x64 patch and 3 contiguous axial slices
            if (cz - 1 < 0) or (cz + 1 >= vol.shape[2]) or \
               (cy - 32 < 0) or (cy + 32 > vol.shape[0]) or \
               (cx - 32 < 0) or (cx + 32 > vol.shape[1]):
                continue

            s0 = normalize_hu(vol[cy-32:cy+32, cx-32:cx+32, cz-1])
            s1 = normalize_hu(vol[cy-32:cy+32, cx-32:cx+32, cz])
            s2 = normalize_hu(vol[cy-32:cy+32, cx-32:cx+32, cz+1])
            patch_3d = np.stack([s0, s1, s2], axis=0)

            # Consensus mask via 50% majority agreement
            try:
                cmask, cbbox, _ = consensus(nodule, clevel=0.5)
                full_mask = np.zeros(vol.shape, dtype=np.float32)
                full_mask[cbbox] = cmask.astype(np.float32)
                patch_mask = full_mask[cy-32:cy+32, cx-32:cx+32, cz]
            except Exception:
                patch_mask = np.zeros((64, 64), dtype=np.float32)

            patch_fname = f"{patient_id}_nodule_{n_idx+1:03d}.npz"
            save_path = out_dir / patch_fname
            np.savez_compressed(save_path, image=patch_3d, mask=patch_mask)

            records.append({
                "subject_id": patient_id,
                "nodule_id": f"nodule_{n_idx+1:03d}",
                "patch_file": str(save_path.resolve()),
                "malignancy_mean": round(mean_mal, 2),
                "label_name": label_name,
                "binary_label": binary_label,
                "center_x": cx,
                "center_y": cy,
                "center_z": cz
            })

    if not records:
        print("[ERROR] No nodules were extracted.")
        return

    df = pd.DataFrame(records)

    # Subject-level split (70/15/15)
    unique_pts = np.array(df["subject_id"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_pts)

    n_tot = len(unique_pts)
    n_tr = int(0.70 * n_tot)
    n_va = int(0.15 * n_tot)

    tr_pts = set(unique_pts[:n_tr])
    va_pts = set(unique_pts[n_tr:n_tr+n_va])

    def assign_split(pid):
        if pid in tr_pts:
            return "train"
        elif pid in va_pts:
            return "val"
        return "test"

    df["split"] = df["subject_id"].apply(assign_split)
    csv_path = out_dir / "large_manifest.csv"
    df.to_csv(csv_path, index=False)

    print("=" * 65)
    print(f"EXTRACTION COMPLETE: {len(df)} total nodules processed.")
    print(f"Manifest written to: {csv_path.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()