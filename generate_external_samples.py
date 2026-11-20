from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def create_synthetic_lung_patch(nodule_type="benign", noise_level=0.08):
    """
    Generates realistic 2.5D normalized CT patches (3, 64, 64) with [-1000, 400] HU profile.
    Lung parenchyma: Low intensity (~0.1 - 0.25)
    Benign nodule: Smooth, circumscribed solid mass (~0.75 - 0.90)
    Malignant nodule: Spiculated, lobulated, heterogeneous mass with halo
    Indeterminate: Small, faint ground-glass / semi-solid border
    Healthy: Vascular lung texture with no mass
    """
    slices = []
    
    for z_offset in [-1, 0, 1]:
        # Background lung parenchyma (-800 to -600 HU normalized -> 0.14 to 0.28)
        base = np.random.normal(0.20, noise_level, (64, 64)).astype(np.float32)
        base = np.clip(base, 0.05, 0.35)

        # Draw pulmonary vessel streaks
        cv2.line(base, (5, 12), (58, 50), 0.35, 1)
        cv2.line(base, (10, 55), (45, 10), 0.30, 1)

        center_y, center_x = 32, 32

        if nodule_type == "benign":
            # Well-circumscribed, round calcified/solid nodule
            radius = int(10 - abs(z_offset) * 2)
            cv2.circle(base, (center_x, center_y), radius, 0.82, -1)
            base = cv2.GaussianBlur(base, (3, 3), 0.8)

        elif nodule_type == "malignant":
            # Large, irregular, spiculated lesion
            radius = int(14 - abs(z_offset) * 2)
            cv2.circle(base, (center_x, center_y), radius, 0.88, -1)
            # Add irregular spiculation spikes
            spikes = [(32, 12), (50, 28), (46, 48), (20, 50), (14, 30), (22, 18)]
            for sx, sy in spikes:
                cv2.line(base, (center_x, center_y), (sx, sy), 0.78, 2)
            base = cv2.GaussianBlur(base, (5, 5), 1.2)

        elif nodule_type == "indeterminate":
            # Ground-glass opacity (faint, borderline morphology)
            radius = int(8 - abs(z_offset) * 2)
            cv2.circle(base, (center_x, center_y), radius, 0.52, -1)
            base = cv2.GaussianBlur(base, (7, 7), 1.6)

        elif nodule_type == "healthy":
            # Normal lung parenchyma with no nodule
            base = cv2.GaussianBlur(base, (3, 3), 0.5)

        slices.append(np.clip(base, 0.0, 1.0))

    patch_3d = np.stack(slices, axis=0).astype(np.float32)
    return patch_3d

def main():
    out_dir = Path("data/external_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("external_case_01_benign", "benign"),
        ("external_case_02_malignant", "malignant"),
        ("external_case_03_indeterminate", "indeterminate"),
        ("external_case_04_normal_lung", "healthy"),
    ]

    print("=" * 65)
    print("GENERATING SYNTHETIC EXTERNAL CT SAMPLES")
    print("=" * 65)

    for name, ntype in samples:
        patch_3d = create_synthetic_lung_patch(nodule_type=ntype)
        
        # Save as .npz (3D patch)
        npz_path = out_dir / f"{name}.npz"
        np.savez_compressed(npz_path, image=patch_3d, mask=np.zeros((64, 64), dtype=np.float32))

        # Save central slice as .png for direct image upload
        png_path = out_dir / f"{name}.png"
        img_uint8 = (patch_3d[1] * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(png_path)

        print(f"[CREATED] {ntype.upper():<14} -> {npz_path.name} & {png_path.name}")

    print("=" * 65)
    print(f"Files saved in: {out_dir.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()