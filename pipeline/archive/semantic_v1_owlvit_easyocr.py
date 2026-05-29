"""
semantic.py — Symbol detection and scoring for segmented marks.

For each mark in <project>/data/segmented/ this script:
  1. Loads the isolated RGBA crop, composited on white
  2. Runs OWL-ViT: absolute detection per symbol — YES/NO + confidence + bounding box
  3. Feeds detected boxes to SAM2 for precise pixel-level masks
  4. Runs CLIP: relative + sigmoid semantic scores across all 32 symbols
  5. Computes overall semantic richness score
  6. Writes all scores to <project>/data/semantic_scores.csv
  7. Saves coloured mask overlays per detected symbol
  8. Saves composite visualisation showing all detections

Usage:
    conda activate othercodes
    python3 pipeline/semantic.py --project /path/to/project
    python3 pipeline/semantic.py          # uses repo root as project
    python3 pipeline/semantic.py --project ~/proj --stems IMG_0001 IMG_0002
    python3 pipeline/semantic.py --threshold 0.15  # adjust detection sensitivity
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
SEG_DIR = RAS_DIR = SEM_DIR = CSV_SEM = MODEL_PATH = None

def _init_paths(project_root: Path):
    global SEG_DIR, RAS_DIR, SEM_DIR, CSV_SEM, MODEL_PATH
    SEG_DIR    = project_root / "data" / "segmented"
    RAS_DIR    = project_root / "data" / "rasters"
    SEM_DIR    = project_root / "data" / "semantic"
    CSV_SEM    = project_root / "data" / "semantic_scores.csv"
    MODEL_PATH = CODE_ROOT / "models" / "sam2_hiera_large.pt"
    SEM_DIR.mkdir(parents=True, exist_ok=True)


# ── 32 prehistoric symbols ────────────────────────────────────────────────────
# (key, display_name, owlvit_query, clip_description)
#
# owlvit_query:   short concrete visual phrase — what the detector looks for
# clip_description: richer description — what CLIP scores semantically

SYMBOLS = [
    # 1. Simple, Foundational Marks
    ("dot",            "Dot",
     "a small filled dot or tiny painted circle",
     "a tiny painted dot or small filled circle mark on a surface"),

    ("cupule",         "Cupule",
     "a small circular cup-shaped hollow or carved depression",
     "a cupule: a small cup-like carved depression or dimple pecked into stone"),

    ("line",           "Line",
     "a single straight engraved or painted line",
     "a single straight line stroke, painted or engraved"),

    ("curved_line",    "Curved Line",
     "a curved or gently arching line stroke",
     "a curved or arcing line, neither straight nor a full circle"),

    ("arch",           "Arch",
     "a semicircular arch or rainbow curve: a smooth symmetric arc that rises from two base points and curves overhead",
     "an arch or rainbow arc: a smooth symmetric upward-curving arc that starts and ends at roughly the same level, like a doorway arch, bridge arch, or rainbow — more regular and symmetric than a general curved line"),

    # 2. Angular Shapes
    ("triangle",       "Triangle",
     "a triangle or three-sided geometric shape",
     "a triangular shape with three sides and three corners"),

    ("square",         "Square / Rectangle",
     "a square or rectangular four-sided outline",
     "a four-sided square or rectangular geometric form"),

    ("cruciform",      "Cruciform",
     "a Christian cross, plus sign, or T-shape: two straight lines intersecting at right angles",
     "a cruciform: two lines crossing at right angles — the Christian cross, the + symbol, or a T-shape, all made from a single horizontal line intersecting a vertical one"),

    ("open_angle",     "Open-Angle",
     "a V shape or chevron or open angular notch",
     "an open-angle or V-shape: two lines meeting at a point like a chevron or arrowhead"),

    ("crosshatch",     "Crosshatch",
     "overlapping crisscrossing parallel lines forming a grid",
     "a crosshatch pattern of intersecting lines forming a grid or diamond lattice"),

    # 3. Complex / Compound Shapes
    ("cordiform",      "Cordiform",
     "a playing-card heart icon or cartoon heart symbol: two symmetrical rounded lobes at the top meeting at a downward-pointing tip",
     "a cordiform or heart symbol: the classic playing-card or cartoon heart shape — two symmetrical rounded lobes at the top converging to a single downward-pointing tip at the base"),

    ("claviform",      "Claviform",
     "a lollipop or tadpole shape: a long narrow straight stem with a single rounded or oval head at one end",
     "a claviform or lollipop shape: a single long narrow stem topped by one rounded bulging head — like a lollipop, a balloon on a string, or a tadpole"),

    ("tectiform",      "Tectiform",
     "a simple house silhouette or tent outline: a triangular peaked roof sitting above a rectangular or square base",
     "a tectiform: the classic simple house silhouette — a triangular peaked roof above a rectangular body with a straight baseline, like a child's drawing of a house or a tent"),

    ("penniform",      "Penniform",
     "a feather or leaf shape with branching lines along a central spine",
     "a penniform: a feather-like shape with barbs or branches extending from a central line like a spine"),

    ("scalariform",    "Scalariform",
     "a ladder with horizontal rungs between two vertical parallel lines",
     "a scalariform or ladder-shaped design: two parallel lines connected by multiple horizontal rungs"),

    ("zigzag",         "Zig-Zag",
     "a zig-zag line with sharp repeated angular back-and-forth changes",
     "a zig-zag: a line that sharply alternates direction repeatedly, like a lightning bolt or saw blade"),

    # 4. Grouped Variations
    ("circle",         "Circle",
     "a circle or circular ring outline, roughly equal in width and height",
     "a full outlined circle or circular ring: a closed round shape with equal width and height, drawn or engraved"),

    ("oval",           "Oval / Ellipse",
     "an oval or ellipse outline, a closed rounded shape that is clearly longer in one direction than the other",
     "an oval or ellipse: a closed rounded shape noticeably longer in one axis than the other, like a stretched circle or egg outline"),

    ("asterisk",       "Asterisk",
     "an asterisk or splat symbol: six or more short lines radiating from a single centre point, like the * character or a snowflake",
     "an asterisk or starburst: six or more lines radiating outward from a single centre, like the * character — distinct from a five-pointed star"),

    ("star",           "Star",
     "a five-pointed star shape with five sharp triangular points arranged symmetrically",
     "a five-pointed star: five triangular points arranged symmetrically around a central pentagon, like the ★ symbol"),

    ("spiral",         "Spiral",
     "a spiral or coiled winding line curving inward or outward",
     "a spiral: a line that continuously curves around a centre, winding inward or outward"),

    ("meander",        "Meander",
     "a Greek key or meander pattern: a continuous angular line that folds back on itself repeatedly in right-angle turns",
     "a meander or Greek key pattern: a continuously folding angular line turning at right angles, like the Greek key border motif or a squared-off labyrinth path"),

    ("branching",      "Branching / Tree",
     "a tree silhouette or branching fork: a central trunk or stem with multiple branches splitting off at angles like a bare winter tree",
     "a branching or tree silhouette: a central trunk or stem with lines forking outward and upward repeatedly, like a bare tree, river delta, or forked lightning"),

    # 5. Hand Gestures & Markings
    ("hand_negative",  "Negative Hand",
     "a clear silhouette outline of a human hand with five spread fingers, made by blowing pigment around a pressed hand against a surface",
     "a negative hand stencil: the precise outline of a human hand showing all five spread fingers, created by spraying pigment around a hand pressed flat"),

    ("hand_positive",  "Positive Hand",
     "a solid painted human handprint with a clearly visible palm and all five fingers stamped directly onto a surface",
     "a positive handprint: a complete human hand covered in paint and pressed flat, showing a solid filled palm and five finger impressions"),

    ("thumb",          "Thumb Stencil",
     "the stencilled outline of a single isolated human thumb or finger, narrow and rounded at the tip",
     "a thumb stencil: the isolated outline of a single human thumb or finger, much smaller and narrower than a full hand"),

    ("finger_fluting", "Finger Fluting",
     "a set of parallel sinuous wavy lines made by dragging multiple fingers side by side through clay or pigment",
     "finger fluting: a group of parallel sinuous channels drawn simultaneously by dragging fingers, with regular spacing matching finger width"),

    ("partial_hand",   "Partial Hand",
     "an incomplete hand stencil or handprint clearly showing a palm but missing one or more amputated or hidden fingers",
     "a partial hand stencil: a hand outline in which one or more fingers are visibly absent, truncated, or folded — otherwise resembling a complete hand"),

    # 6. Figurative & Directional
    ("arrow",          "Arrow",
     "an arrow symbol with a pointed arrowhead at one end and a straight shaft or tail",
     "an arrow: a directional symbol with a pointed triangular or chevron tip at one end and a straight shaft or tail at the other"),

    ("arrowhead",      "Arrowhead",
     "a triangular arrowhead or chevron point: just the pointed tip of an arrow with no shaft",
     "an arrowhead: only the pointed triangular or chevron tip of an arrow, with no shaft or tail — a simple V-shape or triangle pointing in one direction"),

    ("face",           "Face",
     "a face or facial outline showing eyes and a mouth, whether human, animal, or mask-like",
     "a face or facial symbol: a simplified human or animal face showing recognisable features — eyes, nose, or mouth — whether realistic, schematic, or mask-like"),

    # 7. Specialised & Rare Motifs
    ("flabelliform",   "Flabelliform",
     "an open handheld fan or peacock tail: a shape that fans out from a narrow base into a wide semicircular spread of radiating lines",
     "a flabelliform or fan shape: a symbol that spreads outward from a narrow point or base into a wide semicircular arc, like an open folding fan, a peacock tail in display, or a ginkgo leaf"),

    ("reniform",       "Reniform",
     "a kidney or bean shape, an asymmetric oval with an indented side",
     "a reniform: a kidney or bean-shaped symbol, an oval with one concave side"),

    ("serpentiform",   "Serpentiform",
     "a snake or serpent shape: a sinuously curving S-shaped or winding elongated body",
     "a serpentiform or snake shape: an elongated sinuously winding form with S-curves or wave-like bends, like a snake in motion or a winding river — longer and more flowing than a spiral"),

    ("wheel",          "Wheel / Concentric",
     "a sun symbol or wheel: a circle with lines radiating outward like sunrays, or multiple concentric rings",
     "a sun symbol, wheel, or concentric circle motif: a circle with radiating lines extending outward like sunrays, or several rings sharing the same centre — a universally recurring prehistoric sun or wheel motif"),

    ("trilobite",      "Trilobite",
     "an oval shape divided into three equal horizontal sections or lobes, like a segmented pill bug or beetle seen from above",
     "a trilobite-like symbol: an oval or rounded shape clearly divided into three distinct horizontal bands or lobes, like a segmented insect or pill bug viewed from above — three parallel sections within one outline"),

    ("phytomorph",     "Phytomorph",
     "a leaf outline or fern frond: a single leaf shape or a stem with symmetrical leaf-like projections on either side",
     "a phytomorph or plant motif: a recognisable leaf outline, fern frond, or plant stem with leaf-like projections — any organic shape clearly derived from a plant or leaf form"),

    ("geometric",      "Geometric Complex",
     "a complex combination of multiple geometric shapes and lines",
     "a geometric complex: a unique combination of multiple geometric elements, lines, and shapes forming a compound symbol"),
]

# Distinct colour per symbol for visualisation (one per entry in SYMBOLS)
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#800000", "#fffac8",
    "#ff4500", "#00ced1", "#ff69b4", "#7fff00", "#8b0000",
    "#4682b4", "#d2691e", "#32cd32", "#9400d3", "#ff8c00",
    "#20b2aa", "#b8860b", "#6495ed", "#ff1493", "#2e8b57",
    "#c0392b", "#1abc9c", "#e74c3c",
]
SYM_COLORS = {s[0]: _PALETTE[i] for i, s in enumerate(SYMBOLS)}

DETECT_THRESHOLD = 0.20   # OWL-ViT confidence — above this = YES


# ── Device ────────────────────────────────────────────────────────────────────

def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()


# ── Globals — models loaded once ──────────────────────────────────────────────
owlvit_model = owlvit_processor = None
clip_model   = clip_processor   = None
sam_predictor = None
ocr_reader    = None

# EasyOCR language codes to load by default.
# Each code maps to a script family — loading more is slower but covers more marks.
# Latin covers English, French, Spanish, German, Portuguese, Italian, etc.
# Full list: https://www.jaided.ai/easyocr/
OCR_LANGUAGES = ["en", "ar", "ch_sim", "ru"]   # Latin, Arabic, Chinese, Cyrillic


def load_models():
    global owlvit_model, owlvit_processor
    global clip_model, clip_processor
    global sam_predictor

    from transformers import (
        OwlViTProcessor, OwlViTForObjectDetection,
        CLIPProcessor, CLIPModel,
    )

    # OWL-ViT — CPU only (MPS support is incomplete for this model)
    print("Loading OWL-ViT (google/owlvit-base-patch32)...", end=" ", flush=True)
    owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owlvit_model     = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32").to("cpu")
    owlvit_model.eval()
    print("✓")

    # CLIP — use best device
    print(f"Loading CLIP (openai/clip-vit-large-patch14, {DEVICE})...", end=" ", flush=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model     = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14").to(DEVICE)
    clip_model.eval()
    print("✓")

    # EasyOCR — actual text recognition across multiple scripts
    global ocr_reader
    print(f"Loading EasyOCR ({', '.join(OCR_LANGUAGES)})...", end=" ", flush=True)
    try:
        import easyocr
        gpu = DEVICE in ("cuda", "mps")
        ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=gpu, verbose=False)
        print("✓")
    except ImportError:
        print("WARNING: easyocr not installed — text recognition disabled")
        print("  Install with: pip install easyocr")
        ocr_reader = None
    except Exception as e:
        print(f"WARNING: EasyOCR failed to load ({e}) — text recognition disabled")
        ocr_reader = None

    # SAM2 — for precise masks from detected boxes
    if MODEL_PATH and MODEL_PATH.exists():
        print(f"Loading SAM2 ({DEVICE})...", end=" ", flush=True)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            CONFIG    = "configs/sam2.1/sam2.1_hiera_l.yaml"
            _sam2     = build_sam2(CONFIG, str(MODEL_PATH), device=DEVICE)
            sam_predictor = SAM2ImagePredictor(_sam2)
            print("✓")
        except Exception as e:
            print(f"WARNING: SAM2 unavailable ({e}) — using box masks")
            sam_predictor = None
    else:
        print("  SAM2 weights not found — using box masks")


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_mark_rgb(stem: str) -> tuple:
    """Load isolated RGBA crop and composite onto white background → (rgb, mark_mask).

    mark_mask is a boolean H×W array that is True only on actual mark pixels.
    Coloring symbol overlays through this mask prevents background pixels being tinted.

    Returns (rgb_array, bool_mask) or (None, None) if no image found.
    """
    # Preferred: post-analyze.py cropped outputs (mask_crop + isolated_crop)
    crop_path = RAS_DIR  / f"{stem}_isolated_crop.png"
    mask_path = RAS_DIR  / f"{stem}_mask_crop.png"
    if crop_path.exists():
        img = Image.open(crop_path).convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        rgb = np.array(bg.convert("RGB"))
        if mask_path.exists():
            # mask_crop is a grayscale binary: white (255) = mark, black (0) = background
            mark_mask = np.array(Image.open(mask_path).convert("L")) > 127
        else:
            # Fallback: use alpha channel of the isolated crop itself
            mark_mask = np.array(img.split()[3]) > 127
        return rgb, mark_mask

    # Fallback: full-resolution segmented image (pre-analyze.py)
    full_path = SEG_DIR  / f"{stem}_isolated.png"
    if full_path.exists():
        img = Image.open(full_path).convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        rgb       = np.array(bg.convert("RGB"))
        mark_mask = np.array(img.split()[3]) > 127
        return rgb, mark_mask

    return None, None


# ── OWL-ViT detection ─────────────────────────────────────────────────────────

def run_owlvit(img_rgb: np.ndarray, threshold: float) -> list[dict]:
    """
    Run OWL-ViT with all 32 symbol queries.
    Returns list[dict] — one per symbol:
        key, display, score (float), detected (bool), box ([x1,y1,x2,y2] pixels)
    """
    pil   = Image.fromarray(img_rgb)
    h, w  = img_rgb.shape[:2]
    texts = [[s[2] for s in SYMBOLS]]          # one list of queries for one image

    with torch.no_grad():
        inputs  = owlvit_processor(text=texts, images=pil, return_tensors="pt",
                                   padding=True, truncation=True)
        outputs = owlvit_model(**inputs)

    # Manual post-processing — avoids transformers version API differences.
    # outputs.logits:     [1, num_boxes, num_queries]
    # outputs.pred_boxes: [1, num_boxes, 4]  (cx, cy, w, h — normalised 0-1)
    logits   = outputs.logits[0]     # [num_boxes, num_queries]
    pred_boxes = outputs.pred_boxes[0]  # [num_boxes, 4]

    # Per-box, per-query confidence via sigmoid (independent scores)
    scores = torch.sigmoid(logits)   # [num_boxes, num_queries]

    # Convert boxes from (cx,cy,w,h) normalised → (x1,y1,x2,y2) pixels
    cx, cy, bw, bh = (pred_boxes[:, i] for i in range(4))
    x1 = ((cx - bw / 2) * w).clamp(0, w)
    y1 = ((cy - bh / 2) * h).clamp(0, h)
    x2 = ((cx + bw / 2) * w).clamp(0, w)
    y2 = ((cy + bh / 2) * h).clamp(0, h)
    boxes_px = torch.stack([x1, y1, x2, y2], dim=1)  # [num_boxes, 4] pixels

    detections = []
    for sym_idx, sym in enumerate(SYMBOLS):
        sym_scores = scores[:, sym_idx]          # [num_boxes]
        best_idx   = int(sym_scores.argmax())
        best_score = float(sym_scores[best_idx])
        best_box   = boxes_px[best_idx].tolist()

        detections.append({
            "key":      sym[0],
            "display":  sym[1],
            "score":    round(best_score, 4),
            "detected": best_score >= threshold,
            "box":      [round(v, 1) for v in best_box],
        })

    return detections


# ── SAM2 mask from bounding box ───────────────────────────────────────────────

def _box_mask(h: int, w: int, box: list[float]) -> np.ndarray:
    """Fallback rectangular mask when SAM2 is unavailable."""
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in box)
    mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return mask


def mask_from_box(img_rgb: np.ndarray, box: list[float]) -> np.ndarray:
    """Return boolean mask (H×W) for the region containing the detected symbol."""
    h, w = img_rgb.shape[:2]
    if sam_predictor is None:
        return _box_mask(h, w, box)
    try:
        sam_predictor.set_image(img_rgb)
        masks, _, _ = sam_predictor.predict(
            box=np.array(box)[None],
            multimask_output=False,
        )
        return masks[0]
    except Exception:
        return _box_mask(h, w, box)


# ── CLIP scoring ──────────────────────────────────────────────────────────────

def run_clip(img_rgb: np.ndarray) -> dict[str, dict]:
    """
    Score the mark against all 32 symbol descriptions.
    Returns {key: {clip_abs (sigmoid), clip_rel (softmax)}}

    clip_abs: independent score per symbol (sigmoid) — absolute confidence
    clip_rel: competitive score (softmax over 32) — relative ranking
    """
    pil   = Image.fromarray(img_rgb)
    texts = [s[3] for s in SYMBOLS]

    with torch.no_grad():
        inputs  = clip_processor(
            text=texts, images=pil, return_tensors="pt", padding=True
        ).to(DEVICE)
        outputs = clip_model(**inputs)
        logits  = outputs.logits_per_image[0]           # (32,)
        abs_s   = torch.sigmoid(logits).cpu().numpy()   # independent
        rel_s   = torch.softmax(logits, dim=0).cpu().numpy()   # competitive

    return {
        s[0]: {
            "clip_abs": round(float(abs_s[i]),  4),
            "clip_rel": round(float(rel_s[i]),  4),
        }
        for i, s in enumerate(SYMBOLS)
    }


# ── Semantic richness ─────────────────────────────────────────────────────────

def semantic_richness(detections: list[dict], clip_scores: dict) -> dict:
    n     = len(SYMBOLS)
    n_det = sum(1 for d in detections if d["detected"])
    top   = max((d["score"] for d in detections), default=0.0)

    # Shannon entropy of CLIP relative distribution → how spread the signal is
    probs = np.array([clip_scores[s[0]]["clip_rel"] for s in SYMBOLS])
    probs = np.clip(probs, 1e-9, 1.0)
    entropy      = float(-np.sum(probs * np.log(probs)))
    entropy_norm = round(entropy / math.log(n), 4)     # [0, 1]

    # Richness = blend of detection count fraction and CLIP entropy
    richness = round(0.5 * (n_det / n) + 0.5 * entropy_norm, 4)

    return {
        "n_symbols_detected":  n_det,
        "top_detection_score": round(top, 4),
        "clip_entropy":        round(entropy, 4),
        "semantic_richness":   richness,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_outputs(stem: str, img_rgb: np.ndarray,
                   detections: list[dict],
                   masks: dict[str, np.ndarray],
                   mark_mask: np.ndarray | None = None) -> None:
    """Save per-symbol PNGs and a composite overview.

    mark_mask (H×W bool): if provided, symbol color overlays are restricted to
    pixels where the actual mark is present — prevents background tinting.
    """
    h, w  = img_rgb.shape[:2]
    base  = Image.fromarray(img_rgb).convert("RGBA")

    composite_arr = np.array(base, dtype=np.uint8).copy()

    for det in detections:
        if not det["detected"]:
            continue
        key  = det["key"]
        mask = masks.get(key)
        if mask is None or not mask.any():
            continue

        r, g, b = _hex_to_rgb(SYM_COLORS[key])
        alpha   = 160
        mask    = mask.astype(bool)

        # Restrict coloring to actual mark pixels — prevents background tinting
        if mark_mask is not None:
            mask = mask & mark_mask

        # Per-symbol overlay image
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[mask] = [r, g, b, alpha]
        single = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
        single.convert("RGB").save(SEM_DIR / f"{stem}_{key}.png")

        # Accumulate into composite (simple alpha blend over existing)
        mask = mask.astype(bool)
        existing_alpha = composite_arr[:, :, 3].astype(np.float32) / 255.0
        new_alpha      = alpha / 255.0
        out_alpha      = new_alpha + existing_alpha * (1 - new_alpha)
        for c, val in enumerate([r, g, b]):
            composite_arr[:, :, c] = np.where(
                mask,
                np.clip(
                    (val * new_alpha + composite_arr[:, :, c] * existing_alpha * (1 - new_alpha))
                    / np.where(out_alpha > 0, out_alpha, 1),
                    0, 255
                ).astype(np.uint8),
                composite_arr[:, :, c],
            )
        composite_arr[:, :, 3] = np.where(
            mask, np.clip(out_alpha * 255, 0, 255).astype(np.uint8),
            composite_arr[:, :, 3]
        )

    # Label detected symbols
    comp_img = Image.fromarray(composite_arr, "RGBA")
    draw     = ImageDraw.Draw(comp_img)
    detected_labels = [
        d["display"] for d in detections if d["detected"]
    ]
    label = ", ".join(detected_labels) if detected_labels else "no symbols detected"
    draw.text((6, 6),  label, fill=(0,   0,   0, 230))
    draw.text((5, 5),  label, fill=(255, 255, 255, 230))   # shadow

    comp_img.convert("RGB").save(SEM_DIR / f"{stem}_composite.png")


# ── OCR — actual text recovery ────────────────────────────────────────────────

# Script detection heuristics based on Unicode character ranges
def _detect_script(text: str) -> str:
    """Return dominant script name for a string of recovered text."""
    counts = {"Latin": 0, "Arabic": 0, "CJK": 0, "Cyrillic": 0, "Other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x024F:               counts["Latin"]    += 1
        elif 0x0600 <= cp <= 0x06FF:             counts["Arabic"]   += 1
        elif 0x4E00 <= cp <= 0x9FFF:             counts["CJK"]      += 1
        elif 0x3040 <= cp <= 0x30FF:             counts["CJK"]      += 1  # Hiragana/Katakana
        elif 0xAC00 <= cp <= 0xD7AF:             counts["CJK"]      += 1  # Hangul
        elif 0x0400 <= cp <= 0x04FF:             counts["Cyrillic"] += 1
        elif ch.isalpha():                        counts["Other"]    += 1
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "Unknown"


def run_ocr(img_rgb: np.ndarray, min_confidence: float = 0.3) -> dict:
    """
    Run EasyOCR on the mark image and return recovered text.

    Returns a dict with:
        ocr_text        — all detected text joined into one string
        ocr_words       — pipe-separated list of individual detected words/regions
        ocr_word_count  — number of detected text regions above min_confidence
        ocr_confidence  — mean confidence of accepted detections (0–1)
        ocr_has_text    — 1 if any text detected above threshold, else 0
        ocr_script      — dominant script: Latin / Arabic / CJK / Cyrillic / Other / None
    """
    empty = {
        "ocr_text": "", "ocr_words": "", "ocr_word_count": 0,
        "ocr_confidence": 0.0, "ocr_has_text": 0, "ocr_script": "None",
    }

    if ocr_reader is None:
        return empty

    try:
        # EasyOCR accepts numpy RGB arrays directly
        results = ocr_reader.readtext(img_rgb, detail=1, paragraph=False)
        # results: list of ([bbox_points], text, confidence)

        accepted = [
            (text.strip(), conf)
            for _, text, conf in results
            if conf >= min_confidence and text.strip()
        ]

        if not accepted:
            return empty

        words      = [t for t, _ in accepted]
        confs      = [c for _, c in accepted]
        full_text  = " ".join(words)
        script     = _detect_script(full_text)

        return {
            "ocr_text":       full_text,
            "ocr_words":      " | ".join(words),
            "ocr_word_count": len(words),
            "ocr_confidence": round(float(np.mean(confs)), 4),
            "ocr_has_text":   1,
            "ocr_script":     script,
        }

    except Exception as e:
        print(f"\n    OCR error: {e}", end=" ")
        return empty


# ── Process one mark ──────────────────────────────────────────────────────────

def process_mark(stem: str, threshold: float) -> dict | None:
    img_rgb, mark_mask = _load_mark_rgb(stem)
    if img_rgb is None:
        print(f"  ✗ No image found for {stem}")
        return None

    print(f"    OWL-ViT...", end=" ", flush=True)
    detections = run_owlvit(img_rgb, threshold)
    detected   = [d["key"] for d in detections if d["detected"]]

    print(f"SAM2 masks...", end=" ", flush=True)
    masks = {}
    for det in detections:
        if det["detected"]:
            masks[det["key"]] = mask_from_box(img_rgb, det["box"])

    print(f"CLIP...", end=" ", flush=True)
    clip_scores = run_clip(img_rgb)

    print(f"OCR...", end=" ", flush=True)
    ocr = run_ocr(img_rgb)

    sem = semantic_richness(detections, clip_scores)

    render_outputs(stem, img_rgb, detections, masks, mark_mask=mark_mask)

    # Flatten to one CSV row
    row = {"stem": stem}
    for det in detections:
        k = det["key"]
        row[f"owlvit_{k}"]          = int(det["detected"])    # binary YES/NO
        row[f"owlvit_{k}_score"]    = det["score"]            # raw confidence
        row[f"clip_{k}_abs"]        = clip_scores[k]["clip_abs"]   # sigmoid
        row[f"clip_{k}_rel"]        = clip_scores[k]["clip_rel"]   # softmax
    row.update(sem)
    row.update(ocr)   # ocr_text, ocr_words, ocr_word_count, ocr_confidence, ocr_has_text, ocr_script

    ocr_summary = f'  OCR: "{ocr["ocr_text"]}"' if ocr["ocr_has_text"] else "  OCR: none"
    print(f"done.  Detected ({len(detected)}): {detected or 'none'}{ocr_summary}")
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Symbol detection + scoring for segmented marks")
    parser.add_argument("--project",   type=Path, default=None,
                        help="Project root directory (contains data/segmented/)")
    parser.add_argument("--stems",     nargs="+", default=None,
                        help="Process only these stems (default: all)")
    parser.add_argument("--threshold", type=float, default=DETECT_THRESHOLD,
                        help=f"OWL-ViT detection threshold (default: {DETECT_THRESHOLD})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip marks that already have a composite PNG")
    parser.add_argument("--languages", nargs="+", default=None,
                        help=f"EasyOCR language codes (default: {OCR_LANGUAGES}). "
                             "E.g. --languages en ar ja ko ru hi. "
                             "See https://www.jaided.ai/easyocr/ for full list.")
    args = parser.parse_args()

    if args.languages:
        OCR_LANGUAGES.clear()
        OCR_LANGUAGES.extend(args.languages)

    repo_root    = Path(__file__).parent.parent
    project_root = args.project.expanduser().resolve() if args.project else repo_root
    _init_paths(project_root)

    print(f"\nProject:   {project_root}")
    print(f"Device:    {DEVICE}")
    print(f"Threshold: {args.threshold}")
    print(f"Symbols:   {len(SYMBOLS)}\n")

    load_models()

    # Collect stems
    stems = sorted({
        p.stem.replace("_mask", "")
        for p in SEG_DIR.glob("*_mask.png")
    })
    if args.stems:
        stems = [s for s in stems if s in args.stems]
    if args.skip_existing:
        stems = [s for s in stems
                 if not (SEM_DIR / f"{s}_composite.png").exists()]

    if not stems:
        print("No marks to process.")
        return

    print(f"Processing {len(stems)} marks...\n")

    rows = []
    for stem in stems:
        print(f"  {stem}")
        row = process_mark(stem, args.threshold)
        if row:
            rows.append(row)

    if not rows:
        print("No results.")
        return

    cols = list(rows[0].keys())
    with open(CSV_SEM, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Semantic scores → {CSV_SEM}")
    print(f"  {len(rows)} marks × {len(cols) - 1} features")
    print(f"  Visualisations → {SEM_DIR}/\n")


if __name__ == "__main__":
    main()
