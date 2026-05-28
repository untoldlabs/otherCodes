"""
annotate.py — Local annotation tool for other.codes

Runs a local web server at http://localhost:5050

Usage:
    python3 tools/annotate.py                            # open project picker
    python3 tools/annotate.py --project ~/projects/foo   # open project directly
    python3 tools/annotate.py --port 5051                # custom port
"""

import argparse
import io
import json
import re
import shutil
import base64
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import subprocess
from flask import Flask, jsonify, request, send_file, send_from_directory, Response, stream_with_context
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch
import sys

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))
from pipeline.classify import train_and_predict


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".othercodes" / "config.json"

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    cfg = {"projects_dir": str(Path.home() / "Documents" / "other_codes_projects")}
    _save_config(cfg)
    return cfg

def _save_config(cfg: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── Project state ─────────────────────────────────────────────────────────────
# All mutable — updated by _open_project() when user selects a project.

PROJECT_ROOT = None
RAW_DIR = ORIG_DIR = EXCL_DIR = ORPHAN_DIR = SEG_DIR = ANNOT_DIR = META_DIR = CROP_DIR = None

def _open_project(root: Path):
    global PROJECT_ROOT, RAW_DIR, ORIG_DIR, EXCL_DIR, ORPHAN_DIR, SEG_DIR, ANNOT_DIR, META_DIR, CROP_DIR
    PROJECT_ROOT = Path(root).expanduser().resolve()
    RAW_DIR    = PROJECT_ROOT / "data" / "raw"
    ORIG_DIR   = PROJECT_ROOT / "data" / "originals"
    EXCL_DIR   = PROJECT_ROOT / "data" / "excluded"
    ORPHAN_DIR = PROJECT_ROOT / "data" / "orphaned"   # derived files with no raw photo
    SEG_DIR    = PROJECT_ROOT / "data" / "segmented"
    ANNOT_DIR  = PROJECT_ROOT / "data" / "annotations"
    META_DIR   = PROJECT_ROOT / "data" / "metadata"
    CROP_DIR   = PROJECT_ROOT / "data" / "crops"
    for d in [RAW_DIR, ORIG_DIR, EXCL_DIR, ORPHAN_DIR, SEG_DIR, ANNOT_DIR, META_DIR, CROP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  Project: {PROJECT_ROOT}")

def require_project(f):
    """Decorator: return 400 if no project is open."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if PROJECT_ROOT is None:
            return jsonify({"error": "No project open. Select a project first."}), 400
        return f(*args, **kwargs)
    return wrapper


# ── Code-level paths (never change) ───────────────────────────────────────────

MODEL_PATH = CODE_ROOT / "models" / "sam2_hiera_large.pt"

app = Flask(__name__, static_folder=str(CODE_ROOT / "tools" / "static"))

@app.errorhandler(Exception)
def handle_any_error(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Route not found"}), 404


# ── Device selection (MPS → CUDA → CPU) ───────────────────────────────────────

def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()
print(f"Device: {DEVICE}")


# ── Load SAM 2 once at startup ─────────────────────────────────────────────────

print("Loading SAM 2 model...", end=" ", flush=True)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2      = build_sam2(CONFIG, str(MODEL_PATH), device=DEVICE)
    predictor = SAM2ImagePredictor(sam2)
    print("✓")
    SAM_AVAILABLE = True
except Exception as e:
    print(f"\nWARNING: SAM 2 not available ({e})")
    SAM_AVAILABLE = False
    predictor = None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_rgb: np.ndarray, params: dict) -> np.ndarray:
    img        = Image.fromarray(img_rgb.astype(np.uint8))
    blur       = float(params.get("blur", 0))
    brightness = float(params.get("brightness", 0))
    contrast   = float(params.get("contrast", 1.0))
    threshold  = params.get("threshold", None)
    invert     = bool(params.get("invert", False))

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if brightness != 0:
        factor = max(0.0, 1.0 + brightness / 100.0)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if invert:
        arr = np.array(img)
        img = Image.fromarray(255 - arr)
    if threshold is not None:
        gray   = np.array(img.convert("L"))
        binary = ((gray > int(threshold)) * 255).astype(np.uint8)
        img    = Image.fromarray(np.stack([binary, binary, binary], axis=-1))

    return np.array(img)


def img_to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def mask_to_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Photo list ────────────────────────────────────────────────────────────────

def get_photos():
    photos = sorted(RAW_DIR.glob("*.jpg"))
    return [{"name": p.name, "stem": p.stem,
             "done": (SEG_DIR / f"{p.stem}_mask.png").exists()}
            for p in photos]


# ── Duplicate helpers ─────────────────────────────────────────────────────────

def _base_stem(stem: str) -> str:
    """Strip trailing zero-padded duplicate suffix (_0001 …) if present."""
    return re.sub(r'_0\d{3}$', '', stem)

def _next_duplicate_stem(base: str) -> str:
    existing = []
    for f in RAW_DIR.glob(f"{base}_*.jpg"):
        m = re.match(rf'^{re.escape(base)}_(\d{{4}})$', f.stem)
        if m:
            existing.append(int(m.group(1)))
    next_n = (max(existing) + 1) if existing else 1
    return f"{base}_{next_n:04d}"


# ── Annotation persistence ────────────────────────────────────────────────────

def _save_annotations(stem, fg_b64, bg_b64, pp, crop,
                      pipeline="rf_only", sam_score=None, lasso=None,
                      importances=None):
    Image.open(io.BytesIO(base64.b64decode(fg_b64))).save(ANNOT_DIR / f"{stem}_fg.png")
    Image.open(io.BytesIO(base64.b64decode(bg_b64))).save(ANNOT_DIR / f"{stem}_bg.png")
    run_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "stem":       stem,
        "updated_at": run_at,
        "preprocess": pp or {},
        "crop":       crop,
        "lasso":      lasso,
        "pipeline":   pipeline,
    }
    if sam_score is not None:
        meta["sam_score"] = round(sam_score, 4)
    if importances is not None:
        meta["rf_importances"] = importances
    (ANNOT_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Append a row to the project-level importances log for later analysis
    if importances is not None:
        _append_importance_log(stem, run_at, pipeline, importances)

    print(f"  ✓ Annotations saved: {stem}  pipeline={pipeline}")


def _append_importance_log(stem, run_at, pipeline, importances):
    """Append one row to data/rf_importances.csv — one row per RF run."""
    import csv
    log_path = PROJECT_ROOT / "data" / "rf_importances.csv"
    # All known groups in a stable column order
    GROUPS = ["colour", "scale_space", "edge_texture", "sam_hint", "roi_spatial"]
    fieldnames = ["stem", "run_at", "pipeline"] + GROUPS
    row = {"stem": stem, "run_at": run_at, "pipeline": pipeline}
    for g in GROUPS:
        row[g] = round(importances.get(g, 0.0), 4)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _raw_image_path(stem: str) -> Path | None:
    """Find the raw image file for a stem — tries common extensions. Returns None if missing."""
    for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"]:
        p = RAW_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _lasso_mask_np(lasso_pts: list, w: int, h: int) -> np.ndarray:
    """Create a uint8 binary mask (0/255) from lasso polygon [[x,y], ...]."""
    from PIL import ImageDraw
    m    = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    draw.polygon([(int(x), int(y)) for x, y in lasso_pts], fill=255)
    return np.array(m)


def _lasso_bbox(lasso_pts: list, w: int, h: int) -> np.ndarray:
    """Return [x1, y1, x2, y2] bounding box of lasso points, clamped to image."""
    xs = [p[0] for p in lasso_pts]
    ys = [p[1] for p in lasso_pts]
    return np.array([
        max(0, min(xs)), max(0, min(ys)),
        min(w, max(xs)), min(h, max(ys)),
    ], dtype=np.float32)


# ── Ingest helpers ────────────────────────────────────────────────────────────

PHOTO_EXTS = {".jpg", ".jpeg", ".heic", ".png", ".tiff", ".tif", ".webp", ".bmp"}

def _ingest_one(src: Path) -> str:
    """Copy/convert one photo into RAW_DIR (and ORIG_DIR for HEICs). Returns status string.

    HEIC  →  ORIG_DIR/{name}  (lossless archive)  +  RAW_DIR/{stem}.jpg  (converted)
    Other →  RAW_DIR/{name}   (straight copy)
    """
    import pillow_heif
    pillow_heif.register_heif_opener()

    stem    = src.stem
    is_heic = src.suffix.lower() == ".heic"

    if is_heic:
        dst_work = RAW_DIR  / f"{stem}.jpg"
        dst_orig = ORIG_DIR / src.name
    else:
        dst_work = RAW_DIR / src.name
        dst_orig = None

    if dst_work.exists():
        return "skipped"

    try:
        if is_heic:
            # Archive original HEIC
            if not dst_orig.exists():
                shutil.copy2(src, dst_orig)
            # Convert to JPG for annotator
            img = Image.open(src).convert("RGB")
            img.save(dst_work, "JPEG", quality=95)
        else:
            shutil.copy2(src, dst_work)

        _extract_and_save_meta(src, stem)
        return "copied"
    except Exception as e:
        print(f"  ✗ Ingest error ({src.name}): {e}")
        return "error"

def _extract_and_save_meta(src: Path, stem: str):
    try:
        from PIL import ExifTags
        img      = Image.open(src)
        raw_exif = img.getexif()
        if not raw_exif:
            return
        meta = {"stem": stem, "source_filename": src.name,
                "ingested_at": datetime.now(timezone.utc).isoformat()}
        for tag_id, value in raw_exif.items():
            tag = ExifTags.TAGS.get(tag_id)
            if tag == "Make":            meta["camera_make"]  = str(value).strip()
            elif tag == "Model":         meta["camera_model"] = str(value).strip()
            elif tag == "DateTimeOriginal": meta["datetime_original"] = str(value)
        gps_ifd = None
        try:    gps_ifd = raw_exif.get_ifd(0x8825)
        except Exception: pass
        if not gps_ifd: gps_ifd = raw_exif.get(34853)
        if gps_ifd and isinstance(gps_ifd, dict):
            from PIL import ExifTags as ET
            gps = {ET.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            def r2f(r):
                try:
                    return float(r) if not isinstance(r, tuple) else r[0]/r[1]
                except Exception: return 0.0
            def dms(v, ref):
                d = r2f(v[0]) + r2f(v[1])/60 + r2f(v[2])/3600
                return round(-d if ref in ("S","W") else d, 7)
            lat, lon = gps.get("GPSLatitude"), gps.get("GPSLongitude")
            if lat and lon:
                meta["gps_latitude"]  = dms(lat, gps.get("GPSLatitudeRef","N"))
                meta["gps_longitude"] = dms(lon, gps.get("GPSLongitudeRef","E"))
        META_DIR.mkdir(parents=True, exist_ok=True)
        (META_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:
        print(f"  ⚠ EXIF error ({src.name}): {e}")


# ── Routes — project management ───────────────────────────────────────────────

@app.route("/")
def index():
    if PROJECT_ROOT is None:
        return send_file(CODE_ROOT / "tools" / "static" / "projects.html")
    return send_file(CODE_ROOT / "tools" / "static" / "annotate.html")

@app.route("/projects")
def projects_page():
    """Always serve the project picker (even when a project is open)."""
    return send_file(CODE_ROOT / "tools" / "static" / "projects.html")

@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(_load_config())

@app.route("/api/config", methods=["POST"])
def api_config_set():
    cfg = _load_config()
    cfg.update(request.json)
    _save_config(cfg)
    return jsonify({"ok": True, "config": cfg})

@app.route("/api/projects", methods=["GET"])
def api_projects_list():
    cfg      = _load_config()
    base     = Path(cfg["projects_dir"]).expanduser()
    projects = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                raw_dir = d / "data" / "raw"
                seg_dir = d / "data" / "segmented"
                n_photos    = len(list(raw_dir.glob("*.jpg")))    if raw_dir.exists()    else 0
                n_segmented = len(list(seg_dir.glob("*_mask.png"))) if seg_dir.exists() else 0
                projects.append({
                    "name":        d.name,
                    "path":        str(d),
                    "n_photos":    n_photos,
                    "n_segmented": n_segmented,
                    "active":      str(d) == str(PROJECT_ROOT),
                })
    return jsonify({"projects": projects, "projects_dir": str(base)})

@app.route("/api/projects", methods=["POST"])
def api_projects_create():
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"error": "Project name required"}), 400
    cfg  = _load_config()
    path = Path(cfg["projects_dir"]).expanduser() / name
    if path.exists():
        return jsonify({"error": f"Project '{name}' already exists"}), 400
    _open_project(path)   # creates all subdirs
    return jsonify({"ok": True, "name": name, "path": str(path)})

@app.route("/api/projects/open", methods=["POST"])
def api_projects_open():
    path = request.json.get("path")
    if not path:
        return jsonify({"error": "path required"}), 400
    _open_project(Path(path))
    photos = get_photos()
    done   = sum(1 for p in photos if p["done"])
    return jsonify({"ok": True, "path": str(PROJECT_ROOT),
                    "n_photos": len(photos), "n_segmented": done})

@app.route("/api/projects/close", methods=["POST"])
def api_projects_close():
    """Clear the active project so the picker is shown at /."""
    global PROJECT_ROOT, RAW_DIR, ORIG_DIR, EXCL_DIR, ORPHAN_DIR, SEG_DIR, ANNOT_DIR, META_DIR, CROP_DIR
    PROJECT_ROOT = RAW_DIR = ORIG_DIR = EXCL_DIR = ORPHAN_DIR = SEG_DIR = ANNOT_DIR = META_DIR = CROP_DIR = None
    return jsonify({"ok": True})


# ── Routes — ingest ───────────────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
@require_project
def api_ingest():
    """Copy photos from a source folder into the project's data/raw/."""
    src_dir = Path(request.json.get("source_dir", "")).expanduser()
    if not src_dir.is_dir():
        return jsonify({"error": f"Not a directory: {src_dir}"}), 400

    photos = sorted(p for p in src_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in PHOTO_EXTS)
    if not photos:
        return jsonify({"error": "No photos found in that folder"}), 400

    counts = {"copied": 0, "skipped": 0, "error": 0}
    for p in photos:
        status = _ingest_one(p)
        counts[status] = counts.get(status, 0) + 1

    print(f"  Ingest: {counts}")
    return jsonify({"ok": True, **counts, "total": len(photos)})


# ── Routes — annotation ───────────────────────────────────────────────────────

@app.route("/api/photos")
@require_project
def api_photos():
    return jsonify(get_photos())

@app.route("/api/image/<filename>")
@require_project
def api_image(filename):
    return send_from_directory(RAW_DIR, filename)

@app.route("/api/mask/<stem>")
@require_project
def api_mask(stem):
    p = SEG_DIR / f"{stem}_mask.png"
    return send_file(p, mimetype="image/png") if p.exists() else (jsonify({"error": "no mask"}), 404)

def _exclude_stem(stem: str, dest_dir: Path | None = None) -> list[str]:
    """Move ALL derived files for a stem to dest_dir (default: data/excluded/).
    Returns list of moved filenames."""
    target = dest_dir if dest_dir is not None else EXCL_DIR
    target.mkdir(parents=True, exist_ok=True)
    moved = []

    def _mv(path: Path):
        if path.exists():
            dest = target / path.name
            if dest.exists():
                dest = target / (path.stem + "_dup" + path.suffix)
            shutil.move(str(path), str(dest))
            moved.append(path.name)

    # Raw photo
    for ext in [".jpg", ".JPG", ".jpeg", ".HEIC", ".heic", ".png"]:
        _mv(RAW_DIR / f"{stem}{ext}")

    # Segmented outputs
    _mv(SEG_DIR  / f"{stem}_mask.png")
    _mv(SEG_DIR  / f"{stem}_isolated.png")

    # Crops
    _mv(CROP_DIR / f"{stem}_photo.jpg")
    _mv(CROP_DIR / f"{stem}_preview.jpg")

    # Annotations (RF brush strokes + SAM mask + JSON)
    _mv(ANNOT_DIR / f"{stem}_fg.png")
    _mv(ANNOT_DIR / f"{stem}_bg.png")
    _mv(ANNOT_DIR / f"{stem}_sam.png")
    _mv(ANNOT_DIR / f"{stem}.json")

    # Metadata (GPS etc.)
    _mv(META_DIR / f"{stem}.json")

    print(f"  Excluded {stem}: {moved}")
    return moved


@app.route("/api/delete", methods=["POST"])
@require_project
def api_delete():
    stem  = request.json["stem"]
    moved = _exclude_stem(stem)
    return jsonify({"ok": True, "moved": moved})

@app.route("/api/duplicate", methods=["POST"])
@require_project
def api_duplicate():
    stem  = request.json["stem"]
    base  = _base_stem(stem)
    src   = RAW_DIR / f"{base}.jpg"
    if not src.exists():
        return jsonify({"error": f"Source photo not found: {src.name}"}), 404
    new_stem = _next_duplicate_stem(base)
    dst      = RAW_DIR / f"{new_stem}.jpg"
    shutil.copy2(src, dst)
    print(f"  Duplicated: {src.name} → {dst.name}")
    meta_src = META_DIR / f"{base}.json"
    meta_dst = META_DIR / f"{new_stem}.json"
    if meta_src.exists():
        try:
            meta = json.loads(meta_src.read_text())
            meta.update({"stem": new_stem, "source_filename": f"{new_stem}.jpg",
                         "duplicated_from": base,
                         "duplicated_at": datetime.now(timezone.utc).isoformat()})
            meta_dst.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"  ⚠ Could not copy metadata: {e}")
    return jsonify({"ok": True, "new_stem": new_stem, "new_name": f"{new_stem}.jpg"})

@app.route("/api/preview", methods=["POST"])
@require_project
def api_preview():
    try:
        data     = request.json
        stem     = data["stem"]
        params   = data.get("preprocess", {})
        img_rgb  = np.array(Image.open(RAW_DIR / f"{stem}.jpg").convert("RGB"))
        result   = preprocess(img_rgb, params)
        return jsonify({"image": img_to_b64(result)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/segment", methods=["POST"])
@require_project
def api_segment():
    if not SAM_AVAILABLE:
        return jsonify({"error": "SAM 2 not loaded"}), 500
    data      = request.json
    stem      = data["stem"]
    fg_points = data.get("fg_points", [])
    bg_points = data.get("bg_points", [])
    crop      = data.get("crop")
    lasso_pts = data.get("lasso")      # [[x,y], ...] freehand polygon
    pp        = data.get("preprocess", {})
    img_rgb   = np.array(Image.open(_raw_image_path(stem)).convert("RGB"))
    h, w      = img_rgb.shape[:2]
    print(f"  SAM segment: {stem}")
    img_for_sam = preprocess(img_rgb, pp)
    predictor.set_image(img_for_sam)

    # Box prompt: prefer lasso bbox, fall back to crop rect
    box_np = None
    if lasso_pts and len(lasso_pts) > 2:
        box_np = _lasso_bbox(lasso_pts, w, h)
        print(f"  Lasso → bbox {box_np.astype(int)}")
    elif crop and crop.get("w", 0) > 10 and crop.get("h", 0) > 10:
        cx, cy = max(0, int(crop["x"])), max(0, int(crop["y"]))
        cw = min(int(crop["w"]), w - cx)
        ch = min(int(crop["h"]), h - cy)
        box_np = np.array([cx, cy, cx+cw, cy+ch], dtype=np.float32)

    points_np = labels_np = None
    if fg_points or bg_points:
        pts = [[int(x), int(y)] for x, y in fg_points] + [[int(x), int(y)] for x, y in bg_points]
        lbs = [1]*len(fg_points) + [0]*len(bg_points)
        points_np = np.array(pts, dtype=np.float32)
        labels_np = np.array(lbs, dtype=np.int32)
    if points_np is None and box_np is None:
        return jsonify({"error": "Need a box, lasso, or at least one point"}), 400
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=points_np, point_labels=labels_np,
            box=box_np, multimask_output=True)
    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8) * 255

    # Post-process: AND with lasso polygon so nothing outside the squiggle survives
    if lasso_pts and len(lasso_pts) > 2:
        poly = _lasso_mask_np(lasso_pts, w, h)
        mask = np.where(poly > 128, mask, 0).astype(np.uint8)

    print(f"  Score: {scores[best]:.3f}")

    # Persist SAM mask so it can be restored when the photo is reloaded
    sam_path = ANNOT_DIR / f"{stem}_sam.png"
    Image.fromarray(mask).save(sam_path)

    # Update sam_score in the annotation JSON if one already exists
    meta_path = ANNOT_DIR / f"{stem}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["sam_score"] = round(float(scores[best]), 4)
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

    return jsonify({"mask": mask_to_b64(mask), "score": float(scores[best])})

@app.route("/api/annotations/<stem>")
@require_project
def api_annotations(stem):
    fg_path   = ANNOT_DIR / f"{stem}_fg.png"
    bg_path   = ANNOT_DIR / f"{stem}_bg.png"
    sam_path  = ANNOT_DIR / f"{stem}_sam.png"
    meta_path = ANNOT_DIR / f"{stem}.json"
    if not fg_path.exists():
        return jsonify({"exists": False})
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    resp = {
        "exists":     True,
        "fg_strokes": base64.b64encode(fg_path.read_bytes()).decode(),
        "bg_strokes": base64.b64encode(bg_path.read_bytes()).decode(),
        "preprocess": meta.get("preprocess", {}),
        "crop":       meta.get("crop"),
        "lasso":      meta.get("lasso"),
        "updated_at": meta.get("updated_at", ""),
        "sam_score":  meta.get("sam_score"),
    }
    if sam_path.exists():
        resp["sam_mask"] = base64.b64encode(sam_path.read_bytes()).decode()
    return jsonify(resp)

@app.route("/api/classify", methods=["POST"])
@require_project
def api_classify():
    try:
        data      = request.json
        stem      = data["stem"]
        fg_b64    = data["fg_strokes"]
        bg_b64    = data["bg_strokes"]
        crop      = data.get("crop")
        lasso_pts = data.get("lasso")      # [[x,y], ...] freehand polygon
        pp        = data.get("preprocess", {})
        sam_b64   = data.get("sam_hint")
        sam_score = data.get("sam_score")
        pipeline  = "sam_then_rf" if sam_b64 else "rf_only"
        print(f"\n  RF classify: {stem}  pipeline={pipeline}")
        img_rgb  = np.array(Image.open(_raw_image_path(stem)).convert("RGB"))
        img_rgb  = preprocess(img_rgb, pp)
        h, w     = img_rgb.shape[:2]
        fg_mask  = np.array(Image.open(io.BytesIO(base64.b64decode(fg_b64))).convert("L"))
        bg_mask  = np.array(Image.open(io.BytesIO(base64.b64decode(bg_b64))).convert("L"))
        print(f"  FG: {(fg_mask>0).sum()} px  BG: {(bg_mask>0).sum()} px")
        sam_hint_np = None
        if sam_b64:
            sam_hint_np = np.array(Image.open(io.BytesIO(base64.b64decode(sam_b64))).convert("L"))

        # Use lasso bbox as box if no explicit crop given
        effective_crop = crop
        if lasso_pts and len(lasso_pts) > 2 and not crop:
            b = _lasso_bbox(lasso_pts, w, h).astype(int)
            effective_crop = {"x": int(b[0]), "y": int(b[1]),
                              "w": int(b[2]-b[0]), "h": int(b[3]-b[1])}
            print(f"  Lasso → crop {effective_crop}")

        # Build ROI mask for distance-from-edge spatial feature.
        # Prefer lasso polygon (exact shape); fall back to bounding box.
        roi_mask_np = None
        if lasso_pts and len(lasso_pts) > 2:
            poly        = _lasso_mask_np(lasso_pts, w, h)
            roi_mask_np = poly
        elif effective_crop and effective_crop.get("w", 0) > 10:
            # Fill bounding box as a rectangular ROI mask
            roi_mask_np = np.zeros((h, w), dtype=np.uint8)
            cx = max(0, int(effective_crop["x"]))
            cy = max(0, int(effective_crop["y"]))
            cw = min(int(effective_crop["w"]), w - cx)
            ch = min(int(effective_crop["h"]), h - cy)
            roi_mask_np[cy:cy+ch, cx:cx+cw] = 255

        # Clip training strokes to lasso polygon so RF only learns from inside
        if lasso_pts and len(lasso_pts) > 2:
            poly    = roi_mask_np  # already computed above
            fg_mask = np.where(poly > 128, fg_mask, 0).astype(np.uint8)
            bg_mask = np.where(poly > 128, bg_mask, 0).astype(np.uint8)

        result, importances = train_and_predict(img_rgb, fg_mask, bg_mask,
                                               box=effective_crop, sam_hint=sam_hint_np,
                                               roi_mask=roi_mask_np)

        # Post-process: clip RF result to lasso polygon
        if lasso_pts and len(lasso_pts) > 2:
            poly   = _lasso_mask_np(lasso_pts, w, h)
            result = np.where(poly > 128, result, 0).astype(np.uint8)

        _save_annotations(stem, fg_b64, bg_b64, pp, crop,
                          pipeline=pipeline, sam_score=sam_score, lasso=lasso_pts,
                          importances=importances)
        return jsonify({"mask": mask_to_b64(result), "pipeline": pipeline})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"RF failed: {str(e)}"}), 500

def _export_crops(stem: str, mask_np: np.ndarray, img_rgb: np.ndarray) -> str:
    """Generate crop + preview JPEGs for one tag. Returns crop_source string."""
    PAD  = 20
    h, w = mask_np.shape
    x1 = y1 = x2 = y2 = None
    crop_source = "mask"

    annot_path = ANNOT_DIR / f"{stem}.json"
    if annot_path.exists():
        try:
            annot      = json.loads(annot_path.read_text())
            lasso_pts  = annot.get("lasso")
            saved_crop = annot.get("crop")
            if lasso_pts and len(lasso_pts) > 2:
                xs = [p[0] for p in lasso_pts]
                ys = [p[1] for p in lasso_pts]
                x1 = max(0,  int(min(xs)) - PAD)
                y1 = max(0,  int(min(ys)) - PAD)
                x2 = min(w,  int(max(xs)) + PAD)
                y2 = min(h,  int(max(ys)) + PAD)
                crop_source = "lasso"
            elif saved_crop and saved_crop.get("w", 0) > 10:
                x1 = max(0, int(saved_crop["x"]) - PAD)
                y1 = max(0, int(saved_crop["y"]) - PAD)
                x2 = min(w, int(saved_crop["x"]) + int(saved_crop["w"]) + PAD)
                y2 = min(h, int(saved_crop["y"]) + int(saved_crop["h"]) + PAD)
                crop_source = "box"
        except Exception as e:
            print(f"  ⚠ Could not read annotation for crop ({stem}): {e}")

    if x1 is None:
        rows = np.any(mask_np > 128, axis=1)
        cols = np.any(mask_np > 128, axis=0)
        if rows.any():
            y1 = max(0, np.argmax(rows) - PAD)
            y2 = min(h, h - np.argmax(rows[::-1]) + PAD)
            x1 = max(0, np.argmax(cols) - PAD)
            x2 = min(w, w - np.argmax(cols[::-1]) + PAD)

    if x1 is not None and x2 > x1 and y2 > y1:
        Image.fromarray(img_rgb[y1:y2, x1:x2]).save(
            CROP_DIR / f"{stem}_photo.jpg", "JPEG", quality=92)
        crop_mask = mask_np[y1:y2, x1:x2]
        crop_arr  = img_rgb[y1:y2, x1:x2].copy().astype(np.float32)
        fg = crop_mask > 128
        crop_arr[fg,  0] = crop_arr[fg,  0] * 0.5
        crop_arr[fg,  1] = crop_arr[fg,  1] * 0.5 + 255 * 0.5
        crop_arr[fg,  2] = crop_arr[fg,  2] * 0.5
        crop_arr[~fg]    = crop_arr[~fg] * 0.35
        Image.fromarray(crop_arr.clip(0, 255).astype(np.uint8)).save(
            CROP_DIR / f"{stem}_preview.jpg", "JPEG", quality=92)
        return crop_source
    return "none"


@app.route("/api/save", methods=["POST"])
@require_project
def api_save():
    data     = request.json
    stem     = data["stem"]
    mask_b64 = data["mask"]
    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
    mask_np  = np.array(mask_img)

    # ── Full-size outputs ─────────────────────────────────────────────────────
    mask_img.save(SEG_DIR / f"{stem}_mask.png")
    img_rgb = np.array(Image.open(_raw_image_path(stem)).convert("RGB"))
    rgba    = np.dstack([img_rgb, mask_np])
    Image.fromarray(rgba, mode="RGBA").save(SEG_DIR / f"{stem}_isolated.png")

    # ── Tight crop outputs ────────────────────────────────────────────────────
    crop_source = _export_crops(stem, mask_np, img_rgb)
    if crop_source != "none":
        h, w = mask_np.shape
        print(f"  ✓ Saved: {stem}  [{crop_source}] → {CROP_DIR.name}/")
    else:
        print(f"  ✓ Saved: {stem}  (no crop region found)")

    return jsonify({"ok": True, "project_path": str(PROJECT_ROOT)})


@app.route("/api/export_all", methods=["POST"])
@require_project
def api_export_all():
    """Re-export all derived files (isolated PNG + crops) from existing masks.

    Also reports orphaned derived files — crops or segmentations whose raw
    photo no longer exists — so the user can exclude them cleanly.
    """
    mask_stems = sorted({
        p.stem.replace("_mask", "")
        for p in SEG_DIR.glob("*_mask.png")
    })

    # Find orphaned derived files — any stem with derived files but no mask
    mask_stem_set = set(mask_stems)
    derived_stems = set()
    derived_stems |= {p.stem.replace("_photo",    "") for p in CROP_DIR.glob("*_photo.jpg")}
    derived_stems |= {p.stem.replace("_isolated", "") for p in SEG_DIR.glob("*_isolated.png")}
    derived_stems |= {p.stem.replace("_preview",  "") for p in CROP_DIR.glob("*_preview.jpg")}

    orphans = sorted(derived_stems - mask_stem_set)
    if orphans:
        print(f"\n  ⚠ Derived files with no mask: {orphans}")

    if not mask_stems:
        return jsonify({"ok": False, "error": "No segmented masks found",
                        "orphans": orphans})

    print(f"\n  Re-export all: {len(mask_stems)} masks")
    done = skipped = 0
    for stem in mask_stems:
        mask_path = SEG_DIR / f"{stem}_mask.png"
        raw_path  = _raw_image_path(stem)
        if not raw_path or not raw_path.exists():
            print(f"  ⚠ No raw photo for {stem} — adding to orphan report")
            orphans.append(stem)
            skipped += 1
            continue
        try:
            mask_np = np.array(Image.open(mask_path).convert("L"))
            img_rgb = np.array(Image.open(raw_path).convert("RGB"))

            # Regenerate isolated RGBA
            rgba = np.dstack([img_rgb, mask_np])
            Image.fromarray(rgba, mode="RGBA").save(SEG_DIR / f"{stem}_isolated.png")

            # Regenerate crops
            src = _export_crops(stem, mask_np, img_rgb)
            print(f"  ✓ {stem}  [{src}]")
            done += 1
        except Exception as e:
            print(f"  ✗ {stem}: {e}")
            skipped += 1

    print(f"  Done: {done} exported, {skipped} skipped\n")
    return jsonify({"ok": True, "done": done, "skipped": skipped, "orphans": orphans})


@app.route("/api/resolve_orphans", methods=["POST"])
@require_project
def api_resolve_orphans():
    """Move derived files for orphaned stems to excluded/ or orphaned/.

    Body: { "stems": ["IMG_2497_0004", ...], "action": "exclude" | "orphan" }
    "exclude" → data/excluded/  (intentionally removed)
    "orphan"  → data/orphaned/  (raw photo missing, keep for reference)
    """
    data   = request.json
    stems  = data.get("stems", [])
    action = data.get("action", "orphan")
    dest   = EXCL_DIR if action == "exclude" else ORPHAN_DIR

    results = {}
    for stem in stems:
        try:
            moved = _exclude_stem(stem, dest_dir=dest)
            results[stem] = {"ok": True, "moved": moved}
            print(f"  → {action}: {stem}  ({len(moved)} files → {dest.name}/)")
        except Exception as e:
            results[stem] = {"ok": False, "error": str(e)}
            print(f"  ✗ {stem}: {e}")

    return jsonify({"ok": True, "results": results, "dest": dest.name})


# ── Pipeline launcher ─────────────────────────────────────────────────────────

@app.route("/pipeline")
def pipeline_page():
    return send_from_directory(str(CODE_ROOT / "tools" / "static"), "pipeline.html")


@app.route("/api/pipeline/run", methods=["GET"])
@require_project
def api_pipeline_run():
    """Stream pipeline subprocess output as Server-Sent Events.

    Query params:
        script  — "analyze" | "cluster"
        k       — int, number of families (cluster only)
        reps    — int, examples per family (cluster only)
        colors  — comma-separated hex, e.g. #FF3EA5,#00B8D9 (cluster only)
        grid    — int, sparse PCA grid columns (cluster only)
    """
    script = request.args.get("script", "analyze")
    project = str(PROJECT_ROOT)

    python  = sys.executable
    scripts = {"analyze": str(CODE_ROOT / "pipeline" / "analyze.py"),
               "cluster": str(CODE_ROOT / "pipeline" / "cluster.py")}
    if script not in scripts:
        return jsonify({"error": f"Unknown script: {script}"}), 400

    cmd = [python, "-u", scripts[script], "--project", project]

    if script == "cluster":
        k = request.args.get("k", "5")
        reps = request.args.get("reps", "3")
        colors = request.args.get("colors", "")
        grid = request.args.get("grid", "")
        cmd += ["--k", k, "--reps", reps]
        if colors.strip():
            cmd += ["--colors", colors.strip()]
        if grid.strip():
            cmd += ["--grid", grid.strip()]

    def generate():
        yield f"data: [start] Running {script}.py\n\n"
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                # Escape any data: prefix to avoid SSE confusion
                safe = line.replace("\n", " ")
                yield f"data: {safe}\n\n"
            proc.wait()
            code = proc.returncode
            if code == 0:
                yield f"data: [done] ✓ {script}.py finished successfully\n\n"
            else:
                yield f"data: [error] {script}.py exited with code {code}\n\n"
        except Exception as e:
            yield f"data: [error] Failed to start {script}.py: {e}\n\n"
        yield "data: [end]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/pipeline/project")
def api_pipeline_project():
    """Return current project info for the pipeline page."""
    if PROJECT_ROOT is None:
        return jsonify({"project": None})
    features_csv = PROJECT_ROOT / "data" / "features.csv"
    mask_count = len(list((PROJECT_ROOT / "data" / "segmented").glob("*_mask.png"))) \
        if (PROJECT_ROOT / "data" / "segmented").exists() else 0
    has_features = features_csv.exists()
    n_features = 0
    if has_features:
        try:
            with open(features_csv) as f:
                n_features = sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass
    return jsonify({
        "project": str(PROJECT_ROOT),
        "name": PROJECT_ROOT.name,
        "mask_count": mask_count,
        "has_features": has_features,
        "n_features": n_features,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="other.codes annotation tool")
    parser.add_argument("--project", type=Path, default=None,
                        help="Open this project directory immediately")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    if args.project:
        _open_project(args.project)
        photos = get_photos()
        done   = sum(1 for p in photos if p["done"])
        print(f"  {len(photos)} photos  ({done} segmented)")
    else:
        cfg = _load_config()
        print(f"  Projects folder: {cfg['projects_dir']}")
        print(f"  Select a project at http://localhost:{args.port}")

    print(f"  http://localhost:{args.port}\n")
    app.run(port=args.port, debug=False)
