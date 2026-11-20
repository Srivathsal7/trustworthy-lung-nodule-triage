from pathlib import Path
import pylidc as pl

print("=" * 65)
print("DEBUGGING SCAN LOADING & DICOM PATH RESOLUTION")
print("=" * 65)

# Check pylidc configuration path
user_cfg = Path.home() / "pylidc.conf"
if user_cfg.exists():
    print(f"pylidc.conf content:\n{user_cfg.read_text().strip()}")
else:
    print("[WARNING] pylidc.conf does not exist in user home directory!")

scans = pl.query(pl.Scan).all()
print(f"\nTotal scans in pylidc database: {len(scans)}")

# Test the first 5 scans
found_dicom = 0
for scan in scans[:10]:
    pid = scan.patient_id
    series_uid = scan.series_instance_uid
    try:
        # Check if pylidc can locate the DICOM path
        dcm_path = scan.get_path_to_dicom_files()
        print(f"[{pid}] Path found: {dcm_path}")
        
        # Test loading volume
        vol = scan.to_volume(verbose=False)
        print(f"  --> Volume loaded successfully! Shape: {vol.shape}")
        
        nods = scan.cluster_annotations()
        print(f"  --> Nodule clusters found: {len(nods)}")
        found_dicom += 1
        break
    except Exception as e:
        print(f"[{pid}] Failed: {type(e).__name__} -> {e}")

print("=" * 65)
if found_dicom == 0:
    print("Diagnosis: pylidc cannot find the .dcm files on disk for these scans.")