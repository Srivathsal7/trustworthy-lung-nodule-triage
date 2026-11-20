import configparser
from pathlib import Path

# 1. Alias SafeConfigParser to ConfigParser for Python 3.12+ compatibility
configparser.SafeConfigParser = configparser.ConfigParser

# 2. Point to the actual folder containing DICOM files
home = Path.home()
config_file = home / "pylidc.conf"

# Verify which folder contains the DICOM series
possible_dirs = [
    home / "Downloads" / "LIDC-IDRI",
    home / "Downloads" / "lidc_idri",
    home / "Downloads"
]

target_dir = None
for p in possible_dirs:
    if p.exists() and any(p.rglob("*.dcm")):
        target_dir = p
        break

if target_dir is None:
    target_dir = home / "Downloads" / "LIDC-IDRI"

cfg_content = f"""[dicom]
path = {target_dir.resolve()}
warn = True
"""

with open(config_file, "w", encoding="utf-8") as f:
    f.write(cfg_content)

print("=" * 65)
print(f"[SUCCESS] Config file written to: {config_file.resolve()}")
print(f"DICOM Directory configured as:   {target_dir.resolve()}")
print("=" * 65)