"""
ingest.py — Import photos into the other.codes pipeline.

Copies ALL image files from a source directory into the project.
The user's originals are NEVER modified — this script only reads from the source.

Folder logic:
  HEIC files  →  data/originals/  (kept as-is)  +  data/raw/{stem}.jpg  (converted)
  Everything else  →  data/raw/  (copied as-is)

data/raw/ is what the annotator sees. data/originals/ is your lossless archive.

Supported formats: JPG, JPEG, PNG, TIFF, TIF, HEIC, WEBP, BMP

EXIF/GPS metadata is extracted and saved to data/metadata/ (gitignored — never committed).
Re-running on the same source is safe (idempotent): already-present files are skipped.

Usage:
    conda activate othercodes
    python3 pipeline/ingest.py /path/to/photos
    python3 pipeline/ingest.py /path/to/photos --dry-run   # preview only
    python3 pipeline/ingest.py /path/to/photos --project /path/to/project
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pillow_heif
from PIL import Image, ExifTags

pillow_heif.register_heif_opener()

PHOTO_EXTS = {".jpg", ".jpeg", ".heic", ".png", ".tiff", ".tif", ".webp", ".bmp"}


# ── EXIF helpers ───────────────────────────────────────────────────────────────

def _rational_to_float(r) -> float:
    try:
        if hasattr(r, "numerator"):
            return float(r)
        if isinstance(r, tuple) and len(r) == 2:
            return r[0] / r[1] if r[1] != 0 else 0.0
        return float(r)
    except Exception:
        return 0.0


def _dms_to_decimal(dms, ref: str) -> float:
    d = _rational_to_float(dms[0])
    m = _rational_to_float(dms[1])
    s = _rational_to_float(dms[2])
    dec = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        dec = -dec
    return round(dec, 7)


def extract_exif(img_path: Path) -> dict:
    """Extract EXIF metadata. Returns a dict safe for JSON serialisation."""
    meta: dict = {}
    try:
        img      = Image.open(img_path)
        raw_exif = img.getexif()
        if not raw_exif:
            return meta

        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id)
            if tag_name == "Make":
                meta["camera_make"] = str(value).strip()
            elif tag_name == "Model":
                meta["camera_model"] = str(value).strip()
            elif tag_name == "DateTimeOriginal":
                meta["datetime_original"] = str(value)
            elif tag_name == "DateTime" and "datetime_original" not in meta:
                meta["datetime_original"] = str(value)
            elif tag_name in ("ImageWidth", "ExifImageWidth"):
                meta.setdefault("width_px", int(value))
            elif tag_name in ("ImageLength", "ExifImageHeight"):
                meta.setdefault("height_px", int(value))

        gps_ifd = None
        try:
            gps_ifd = raw_exif.get_ifd(0x8825)
        except Exception:
            pass
        if not gps_ifd:
            gps_ifd = raw_exif.get(34853)

        if gps_ifd and isinstance(gps_ifd, dict):
            gps     = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            lat_dms = gps.get("GPSLatitude")
            lon_dms = gps.get("GPSLongitude")
            lat_ref = gps.get("GPSLatitudeRef", "N")
            lon_ref = gps.get("GPSLongitudeRef", "E")
            if lat_dms and lon_dms:
                meta["gps_latitude"]  = _dms_to_decimal(lat_dms, lat_ref)
                meta["gps_longitude"] = _dms_to_decimal(lon_dms, lon_ref)
            alt = gps.get("GPSAltitude")
            if alt is not None:
                alt_m = _rational_to_float(alt)
                if gps.get("GPSAltitudeRef") in (1, b"\x01"):
                    alt_m = -alt_m
                meta["gps_altitude_m"] = round(float(alt_m), 1)

    except Exception as e:
        print(f"    ⚠ EXIF error ({img_path.name}): {e}")

    return meta


# ── File helpers ───────────────────────────────────────────────────────────────

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_photos(src: Path) -> list[Path]:
    """Return all recognised image files in src (non-recursive, sorted)."""
    return sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTS
    )


# ── Per-photo ingest ───────────────────────────────────────────────────────────

def ingest_photo(src_path: Path, raw_dir: Path, orig_dir: Path, meta_dir: Path,
                 *, dry_run: bool) -> str:
    """
    Ingest one photo. Returns 'copied', 'skipped', or 'error'.

    HEIC  →  orig_dir/{name}  +  raw_dir/{stem}.jpg  (converted)
    Other →  raw_dir/{name}   (straight copy)
    """
    is_heic = src_path.suffix.lower() == ".heic"
    stem    = src_path.stem

    if is_heic:
        dst_work = raw_dir  / f"{stem}.jpg"   # annotator-visible converted file
        dst_orig = orig_dir / src_path.name   # lossless archive
        label    = f"{src_path.name}  →  {dst_work.name}  (original → originals/)"
    else:
        dst_work = raw_dir  / src_path.name
        dst_orig = None
        label    = src_path.name

    # Already present? Skip if MD5 matches.
    if dst_work.exists():
        if _md5(src_path) == _md5(dst_work) if not is_heic else dst_work.exists():
            print(f"  skip  {src_path.name}")
            return "skipped"

    if dry_run:
        print(f"  [copy]  {label}")
        return "dry_run"

    # Create dirs
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if is_heic:
        # Archive original
        orig_dir.mkdir(parents=True, exist_ok=True)
        if not dst_orig.exists():
            shutil.copy2(src_path, dst_orig)

        # Convert to JPG for raw/
        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dst_work, "JPEG", quality=95)
            print(f"  ✓  {label}")
        except Exception as e:
            print(f"  ✗  HEIC conversion failed ({src_path.name}): {e}")
            return "error"
    else:
        shutil.copy2(src_path, dst_work)
        print(f"  ✓  {label}")

    # EXIF / GPS — read from original source
    exif      = extract_exif(src_path)
    meta_path = meta_dir / f"{stem}.json"
    meta      = {
        "stem":            stem,
        "source_filename": src_path.name,
        "ingested_at":     datetime.now(timezone.utc).isoformat(),
        **exif,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if "gps_latitude" in meta:
        print(f"       GPS: {meta['gps_latitude']:.5f}, {meta['gps_longitude']:.5f}")

    return "copied"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import photos into an other.codes project."
    )
    parser.add_argument("src", type=Path,
                        help="Source directory containing photos to ingest.")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project root (default: repo root — for standalone use).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be ingested without touching anything.")
    args = parser.parse_args()

    project_root = (args.project or Path(__file__).parent.parent).expanduser().resolve()
    raw_dir  = project_root / "data" / "raw"
    orig_dir = project_root / "data" / "originals"
    meta_dir = project_root / "data" / "metadata"

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: Source is not a directory: {src}")
        return

    photos = _find_photos(src)
    if not photos:
        print(f"No image files found in {src}")
        print(f"  Supported: {', '.join(sorted(PHOTO_EXTS))}")
        return

    prefix = "DRY RUN — " if args.dry_run else ""
    print(f"\n  {prefix}Ingesting {len(photos)} file(s) from:")
    print(f"  {src}")
    print(f"  HEIC  →  {orig_dir}  +  converted JPG in {raw_dir}")
    print(f"  Other →  {raw_dir}\n")

    counts: dict[str, int] = {}
    for photo in photos:
        status = ingest_photo(photo, raw_dir, orig_dir, meta_dir, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1

    print()
    if args.dry_run:
        print(f"  Dry run complete — {counts.get('dry_run', 0)} file(s) would be ingested.")
    else:
        copied  = counts.get("copied", 0)
        skipped = counts.get("skipped", 0)
        errors  = counts.get("error", 0)
        print(f"  ✓ {copied} ingested  |  {skipped} already present  |  {errors} errors")
        if copied:
            print(f"\n  Working files  →  {raw_dir}")
            print(f"  HEIC originals →  {orig_dir}")
            print(f"  Metadata       →  {meta_dir}  (gitignored — GPS stays private)")
    print()


if __name__ == "__main__":
    main()
