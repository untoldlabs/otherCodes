"""
convert.py — Convert HEIC photos to JPG for downstream processing.

Usage:
    python pipeline/convert.py
"""

import os
from pathlib import Path
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = RAW_DIR  # JPGs sit alongside HEICs in data/raw/


def convert_heic_to_jpg(heic_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / (heic_path.stem + ".jpg")
    if out_path.exists():
        print(f"  skip (already converted): {out_path.name}")
        return out_path
    img = Image.open(heic_path)
    img = img.convert("RGB")
    img.save(out_path, "JPEG", quality=95)
    print(f"  ✓ {heic_path.name} → {out_path.name}")
    return out_path


def main():
    heic_files = sorted(RAW_DIR.glob("*.HEIC")) + sorted(RAW_DIR.glob("*.heic"))
    if not heic_files:
        print("No HEIC files found in", RAW_DIR)
        return

    print(f"Converting {len(heic_files)} HEIC files...\n")
    for heic in heic_files:
        convert_heic_to_jpg(heic, OUT_DIR)
    print(f"\nDone. JPGs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
