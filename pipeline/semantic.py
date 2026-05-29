"""
semantic.py — CLIP-based semantic scoring for segmented graffiti marks.

For each mark this script:
  1. Loads the isolated RGBA crop, composited on white
  2. Runs CLIP: cosine similarity between the image and each symbol's text description
  3. Computes relative (softmax) and absolute (cosine) scores for all symbols
  4. Computes Shannon entropy of the relative distribution (how concentrated vs spread)
  5. Writes per-mark scores to <project>/data/semantic_scores.csv

History:
  v1 — OWL-ViT + EasyOCR + SAM2 + CLIP.  Archived: pipeline/archive/semantic_v1_owlvit_easyocr.py
       OWL-ViT and EasyOCR produced no useful output on graffiti crops.
       Symbol detection is handled by vlm.py (Qwen2.5-VL).
       This script now provides CLIP feature vectors for PCA / clustering.

Usage:
    conda activate othercodes
    python3 pipeline/semantic.py --project /path/to/project
    python3 pipeline/semantic.py --project ~/proj --stems IMG_0001 IMG_0002
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
SEG_DIR = RAS_DIR = SEM_DIR = CSV_SEM = None

def _init_paths(project_root: Path):
    global SEG_DIR, RAS_DIR, SEM_DIR, CSV_SEM
    SEG_DIR = project_root / "data" / "segmented"
    RAS_DIR = project_root / "data" / "rasters"
    SEM_DIR = project_root / "data" / "semantic"
    CSV_SEM = project_root / "data" / "semantic_scores.csv"
    SEM_DIR.mkdir(parents=True, exist_ok=True)


# ── Symbol list ───────────────────────────────────────────────────────────────
# (key, display_name, clip_description)
# clip_description: rich text phrase used as the CLIP text embedding target.
# Richer, more specific descriptions outperform single-word queries.

SYMBOLS = [
    ("dot",            "Dot",
     "a tiny painted dot or small filled circle mark on a surface"),
    ("cupule",         "Cupule",
     "a cupule: a small cup-like carved depression or dimple pecked into stone"),
    ("line",           "Line",
     "a single straight line stroke, painted or engraved"),
    ("curved_line",    "Curved Line",
     "a curved or arcing line, neither straight nor a full circle"),
    ("arch",           "Arch",
     "an arch or rainbow arc: a smooth symmetric upward-curving arc, like a doorway arch or rainbow"),
    ("triangle",       "Triangle",
     "a triangular shape with three sides and three corners"),
    ("square",         "Square / Rectangle",
     "a four-sided square or rectangular geometric form"),
    ("cruciform",      "Cruciform",
     "a cruciform: two lines crossing at right angles — the Christian cross, the + symbol, or a T-shape"),
    ("open_angle",     "Open-Angle",
     "an open-angle or V-shape: two lines meeting at a point like a chevron or arrowhead"),
    ("crosshatch",     "Crosshatch",
     "a crosshatch pattern of intersecting lines forming a grid or diamond lattice"),
    ("cordiform",      "Cordiform",
     "a cordiform or heart symbol: the classic playing-card heart shape — two rounded lobes converging to a downward point"),
    ("claviform",      "Claviform",
     "a claviform or lollipop shape: a single long narrow stem topped by one rounded bulging head"),
    ("tectiform",      "Tectiform",
     "a tectiform: the classic simple house silhouette — a triangular peaked roof above a rectangular body"),
    ("penniform",      "Penniform",
     "a penniform: a feather-like shape with barbs or branches extending from a central line like a spine"),
    ("scalariform",    "Scalariform",
     "a scalariform or ladder-shaped design: two parallel lines connected by multiple horizontal rungs"),
    ("zigzag",         "Zig-Zag",
     "a zig-zag: a line that sharply alternates direction repeatedly, like a lightning bolt or saw blade"),
    ("circle",         "Circle",
     "a full outlined circle or circular ring: a closed round shape drawn or engraved"),
    ("oval",           "Oval / Ellipse",
     "an oval or ellipse: a closed rounded shape noticeably longer in one axis than the other, like an egg outline"),
    ("asterisk",       "Asterisk",
     "an asterisk or starburst: six or more lines radiating outward from a single centre, like the * character"),
    ("star",           "Star",
     "a five-pointed star: five triangular points arranged symmetrically around a central pentagon, like ★"),
    ("spiral",         "Spiral",
     "a spiral: a line that continuously curves around a centre, winding inward or outward"),
    ("meander",        "Meander",
     "a meander or Greek key pattern: a continuously folding angular line turning at right angles"),
    ("branching",      "Branching / Tree",
     "a branching or tree silhouette: a central trunk with lines forking outward, like a bare tree or river delta"),
    ("hand_negative",  "Negative Hand",
     "a negative hand stencil: the precise outline of a human hand with five spread fingers"),
    ("hand_positive",  "Positive Hand",
     "a positive handprint: a complete human hand in paint pressed flat, showing palm and five fingers"),
    ("thumb",          "Thumb Stencil",
     "a thumb stencil: the isolated outline of a single human thumb or finger"),
    ("finger_fluting", "Finger Fluting",
     "finger fluting: a group of parallel sinuous channels drawn simultaneously by dragging fingers"),
    ("partial_hand",   "Partial Hand",
     "a partial hand stencil: a hand outline with one or more fingers visibly absent or truncated"),
    ("arrow",          "Arrow",
     "an arrow: a directional symbol with a pointed tip at one end and a straight shaft at the other"),
    ("arrowhead",      "Arrowhead",
     "an arrowhead: only the pointed triangular tip of an arrow, with no shaft"),
    ("face",           "Face",
     "a face or facial symbol: a simplified human or animal face showing eyes, nose, or mouth"),
    ("flabelliform",   "Flabelliform",
     "a flabelliform or fan shape: a symbol spreading from a narrow point into a wide semicircular arc"),
    ("reniform",       "Reniform",
     "a reniform: a kidney or bean-shaped symbol, an oval with one concave side"),
    ("serpentiform",   "Serpentiform",
     "a serpentiform or snake shape: an elongated sinuously winding form with S-curves, like a snake"),
    ("wheel",          "Wheel / Concentric",
     "a sun symbol, wheel, or concentric circle motif: a circle with radiating lines or several concentric rings"),
    ("trilobite",      "Trilobite",
     "a trilobite-like symbol: an oval clearly divided into three distinct horizontal bands or lobes"),
    ("phytomorph",     "Phytomorph",
     "a phytomorph or plant motif: a recognisable leaf outline, fern frond, or plant stem with leaf-like projections"),
    ("geometric",      "Geometric Complex",
     "a geometric complex: a unique combination of multiple geometric elements forming a compound symbol"),
]

SYMBOL_KEYS = [s[0] for s in SYMBOLS]


# ── Device ────────────────────────────────────────────────────────────────────

def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()


# ── Model ─────────────────────────────────────────────────────────────────────

clip_model     = None
clip_processor = None
_text_features = None   # cached — same for every mark in a run


def load_clip():
    global clip_model, clip_processor, _text_features
    from transformers import CLIPProcessor, CLIPModel

    print(f"Loading CLIP (openai/clip-vit-large-patch14, {DEVICE})...", end=" ", flush=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_model.eval()
    print("✓")

    # Pre-encode all symbol descriptions — this only runs once
    texts  = [s[2] for s in SYMBOLS]
    inputs = clip_processor(text=texts, return_tensors="pt",
                            padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        _text_features = clip_model.get_text_features(**inputs)   # (N, D)
        _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
    print(f"  Text features cached for {len(SYMBOLS)} symbols.")


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_mark_rgb(stem: str) -> "np.ndarray | None":
    for candidate, is_rgba in [
        (RAS_DIR / f"{stem}_isolated_crop.png", True),
        (SEG_DIR / f"{stem}_isolated.png",      True),
    ]:
        if candidate.exists():
            img = Image.open(candidate).convert("RGBA")
            bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return np.array(bg.convert("RGB"))
    return None


# ── CLIP scoring ──────────────────────────────────────────────────────────────

def run_clip(img_rgb: "np.ndarray") -> dict[str, dict]:
    """
    Score the mark against all symbol descriptions using cosine similarity.

    Returns {key: {"clip_cos": float, "clip_rel": float}}

    clip_cos: raw cosine similarity ∈ [-1, 1], typically [0, 1] for meaningful matches.
              Mathematically grounded — derived from the angle between image and text
              embeddings in CLIP's shared vector space.

    clip_rel: softmax over all N symbols ∈ (0, 1) summing to 1 — relative ranking.
              Tells you which symbols this mark resembles MOST, not in absolute terms.
              Use this for PCA / clustering features.
    """
    pil    = Image.fromarray(img_rgb)
    inputs = clip_processor(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)        # (1, D)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        cos_sim  = (img_feat @ _text_features.T).squeeze(0)      # (N,)
        rel      = torch.softmax(cos_sim * 100, dim=0)            # sharpen then softmax

    cos_np = cos_sim.cpu().numpy()
    rel_np = rel.cpu().numpy()

    return {
        s[0]: {
            "clip_cos": round(float(cos_np[i]), 4),
            "clip_rel": round(float(rel_np[i]), 4),
        }
        for i, s in enumerate(SYMBOLS)
    }


# ── Semantic richness ─────────────────────────────────────────────────────────

def semantic_richness(clip_scores: dict) -> dict:
    """
    Shannon entropy of the CLIP relative distribution.
    High entropy = signal spread across many symbols (complex / multi-element mark).
    Low entropy  = signal concentrated on one symbol (simple / iconic mark).
    """
    n     = len(SYMBOLS)
    probs = np.array([clip_scores[k]["clip_rel"] for k in SYMBOL_KEYS])
    probs = np.clip(probs, 1e-9, 1.0)
    entropy      = float(-np.sum(probs * np.log(probs)))
    entropy_norm = round(entropy / math.log(n), 4)   # [0, 1]

    top_key   = SYMBOL_KEYS[int(np.argmax(probs))]
    top_score = round(float(probs.max()), 4)

    return {
        "top_symbol":        top_key,
        "top_clip_rel":      top_score,
        "clip_entropy":      round(entropy, 4),
        "clip_entropy_norm": entropy_norm,
    }


# ── Process one mark ──────────────────────────────────────────────────────────

def process_mark(stem: str) -> dict | None:
    img_rgb = _load_mark_rgb(stem)
    if img_rgb is None:
        print(f"  ✗ No image found for {stem}")
        return None

    clip_scores = run_clip(img_rgb)
    sem         = semantic_richness(clip_scores)

    row = {"stem": stem}
    for key in SYMBOL_KEYS:
        row[f"clip_cos_{key}"] = clip_scores[key]["clip_cos"]
        row[f"clip_rel_{key}"] = clip_scores[key]["clip_rel"]
    row.update(sem)

    top = sem["top_symbol"]
    print(f"  {stem}  →  top: {top} ({sem['top_clip_rel']:.3f})  "
          f"entropy_norm: {sem['clip_entropy_norm']:.3f}")
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP semantic scoring for segmented graffiti marks")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project root (contains data/rasters/ or data/segmented/)")
    parser.add_argument("--stems",   nargs="+", default=None,
                        help="Process only these stems (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stems already in semantic_scores.csv")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve() if args.project \
                   else Path(__file__).parent.parent
    _init_paths(project_root)

    print(f"\nProject : {project_root}")
    print(f"Device  : {DEVICE}")
    print(f"Symbols : {len(SYMBOLS)}\n")

    load_clip()

    # Collect stems
    stems = sorted({
        p.stem.replace("_isolated_crop", "")
        for p in RAS_DIR.glob("*_isolated_crop.png")
    } | {
        p.stem.replace("_isolated", "")
        for p in SEG_DIR.glob("*_isolated.png")
    })

    if args.stems:
        stems = [s for s in stems if s in args.stems]

    if args.skip_existing and CSV_SEM.exists():
        import csv as _csv
        with open(CSV_SEM) as f:
            done = {row["stem"] for row in _csv.DictReader(f)}
        stems = [s for s in stems if s not in done]

    if not stems:
        print("No marks to process.")
        return

    print(f"\nProcessing {len(stems)} marks...\n")

    rows = []
    for stem in stems:
        row = process_mark(stem)
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
    print(f"  {len(rows)} marks × {len(cols) - 1} features\n")


if __name__ == "__main__":
    main()
