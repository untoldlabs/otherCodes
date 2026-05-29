"""
vlm_core.py — Shared infrastructure for all VLM pipeline scripts.

Imported by vlm_figures.py, vlm_geometry.py, vlm_glyphs.py,
vlm_text.py, and vlm_combined.py.

NOT meant to be run directly — use the individual focused scripts.

Contains:
  - SYMBOLS / SYMBOL_KEYS / SYMBOL_DISPLAY / SYMBOL_KEYWORDS
  - Ollama call helpers (_call_ollama, _check_ollama, _parse_json)
  - Image loading helpers (_image_to_base64, load_mark_image, load_mask_array)
  - Grep + hedge filter (_grep_symbols, _is_hedged)
  - Rendering primitives (_draw_pixel_overlay, _word_wrap, DOT_COLOURS)
  - Stem collection (collect_stems)
"""

import base64
import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE          = "http://localhost:11434"
OLLAMA_URL           = f"{OLLAMA_BASE}/api/chat"
VLM_MODEL            = "qwen2.5vl:7b"
CONFIDENCE_THRESHOLD = 76   # display only; grep scoring starts at 80


# ── Symbol list ───────────────────────────────────────────────────────────────

SYMBOLS = [
    # Geometric primitives
    ("accent_dot",     "Floating accent dot / isolated point"),
    ("arch",           "Arch / rainbow arc"),
    ("zigzag",         "Zig-zag / lightning bolt"),
    ("circle",         "Circle"),
    ("oval",           "Oval / ellipse"),
    ("triangle",       "Triangle"),
    ("square",         "Square / rectangle"),
    ("crosshatch",     "Crosshatch / grid"),
    # Symbolic shapes
    ("cruciform",      "Cross / plus sign"),
    ("open_angle",     "V-shape / chevron"),
    ("asterisk",       "Asterisk / starburst (*)"),
    ("star",           "Five-pointed star"),
    ("spiral",         "Spiral / swirl"),
    ("meander",        "Meander / Greek key"),
    ("branching",      "Branching / tree"),
    ("cordiform",      "Heart / cordiform"),
    ("arrow",          "Arrow (with shaft)"),
    ("arrowhead",      "Arrowhead only (no shaft)"),
    ("wheel",          "Sun symbol / wheel / concentric circles"),
    ("serpentiform",   "Snake / S-curve / serpentiform"),
    ("eye",            "Eye symbol (open or closed, realistic or schematic — single eye)"),
    ("peace",          "Peace symbol (☮)"),
    ("drip",           "Paint drip / teardrop"),
    # Figurative
    ("face",           "Face (any — smiley, schematic, mask)"),
    ("hand_negative",  "Negative hand stencil"),
    ("hand_positive",  "Positive handprint"),
    ("partial_hand",   "Partial hand / missing fingers"),
    # Rare / prehistoric
    ("claviform",      "Claviform / lollipop shape"),
    ("tectiform",      "Tectiform / house outline"),
    ("penniform",      "Penniform / feather"),
    ("scalariform",    "Scalariform / ladder"),
    ("flabelliform",   "Flabelliform / fan shape"),
    ("reniform",       "Reniform / kidney shape"),
    ("trilobite",      "Trilobite / segmented oval"),
    ("phytomorph",     "Phytomorph / leaf or fern"),
    ("cupule",         "Cupule / carved depression"),
    ("finger_fluting", "Finger fluting / parallel wavy lines"),
    ("thumb",          "Thumb stencil"),
    ("geometric",      "Complex geometric / compound symbol"),
]

SYMBOL_KEYS    = [s[0] for s in SYMBOLS]
SYMBOL_DISPLAY = {s[0]: s[1] for s in SYMBOLS}


# ── Symbol keyword grep ────────────────────────────────────────────────────────

SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "accent_dot":    ["dot", "dots", "floating dot", "floating point", "accent", "isolated point",
                      "small mark", "spot", "spots", "point above", "floating mark",
                      "small dot", "tiny dot", "filled dot", "paint dot", "isolated dot",
                      "small circle", "small round", "small filled"],
    "arch":          ["arch", "arc", "rainbow arc", "dome", "curved arch", "arched",
                      "curved line", "curved stroke", "sweeping curve", "arching curve"],
    "zigzag":        ["zigzag", "zig-zag", "lightning", "jagged", "serrated line",
                      "angular line", "sharp angles", "sharp turns", "jagged line"],
    "circle":        ["circle", "circular", "round loop", "enclosed circle", "ring", "disc",
                      "loop", "loops", "closed loop", "rounded loop", "circular loop",
                      "circular shape", "circular outline", "circular form"],
    "oval":          ["oval", "ellipse", "elliptical", "egg shape", "elongated circle",
                      "oblong", "oval loop", "oval shape", "irregular loop", "large loop",
                      "elongated loop", "rounded shape"],
    "triangle":      ["triangle", "triangular", "three-sided", "pyramidal"],
    "square":        ["square", "rectangle", "rectangular", "box shape", "enclosed box"],
    "crosshatch":    ["crosshatch", "grid", "hatching", "cross-hatch", "lattice"],
    "cruciform":     ["cross", "plus sign", "cruciform", "t-shape", "plus-shaped"],
    "open_angle":    ["v-shape", "chevron", "v shape", "open angle", "v-shaped"],
    "asterisk":      ["asterisk", "starburst", "star burst", "radial lines", "asterisk-like"],
    "star":          ["five-pointed star", "five point star", "pentagram", "star shape"],
    "spiral":        ["spiral", "swirl", "coil", "curl", "whirl", "vortex", "spiralling",
                      "spiral loop", "swirling"],
    "meander":       ["meander", "greek key", "labyrinthine", "maze"],
    "branching":     ["branch", "branching", "fork", "forked", "tree-like"],
    "cordiform":     ["heart", "heart shape", "heart-shaped", "cordiform",
                      "love heart", "heart motif", "heart symbol", "heart outline",
                      "heart form", "cardiac", "love symbol"],
    "arrow":         ["arrow", "arrow shape", "arrowhead with shaft", "pointing arrow"],
    "arrowhead":     ["arrowhead", "arrow tip", "pointed tip", "arrow point"],
    "wheel":         ["wheel", "sun symbol", "radial", "spokes", "concentric circle"],
    "serpentiform":  ["snake", "serpent", "s-curve", "serpentine", "sinuous", "s-shaped",
                      "serpentiform", "snake-like", "s shape"],
    "eye":           ["eye", "single eye", "eye symbol", "pupil", "iris", "eye shape"],
    "peace":         ["peace", "peace sign", "peace symbol"],
    "drip":          ["drip", "paint drip", "drop", "teardrop", "dribble", "dripping"],
    "face":          ["face", "facial", "head", "portrait", "smiley", "mask",
                      "eyes and mouth", "nose and mouth", "face-like", "stylized face",
                      "human face", "cartoon face", "schematic face",
                      "humanoid", "figure with", "cartoon character", "character with",
                      "skull", "face shape", "face outline", "face motif",
                      "eyes nose", "eyes mouth", "two eyes", "eye and mouth",
                      "human figure", "person", "caricature", "grimace",
                      "head shape", "head outline", "head motif",
                      "anthropomorphic", "face-shaped"],
    "hand_negative": ["negative hand", "hand stencil", "hand outline"],
    "hand_positive": ["handprint", "hand print", "positive hand", "palm print"],
    "partial_hand":  ["partial hand", "missing finger", "incomplete hand"],
    "claviform":     ["claviform", "lollipop", "lollipop shape", "tadpole"],
    "tectiform":     ["tectiform", "house shape", "tent shape", "roof outline"],
    "penniform":     ["penniform", "feather", "feather shape", "quill"],
    "scalariform":   ["scalariform", "ladder", "ladder shape"],
    "flabelliform":  ["flabelliform", "fan", "fan shape", "open fan"],
    "reniform":      ["reniform", "kidney", "kidney shape"],
    "trilobite":     ["trilobite", "segmented oval"],
    "phytomorph":    ["phytomorph", "leaf", "leaf shape", "fern", "plant shape"],
    "cupule":        ["cupule", "carved depression", "cup mark"],
    "finger_fluting":["finger fluting", "parallel wavy", "finger marks"],
    "thumb":         ["thumb", "thumb stencil"],
    "geometric":     ["complex geometric", "compound symbol", "geometric compound"],
}

# Words indicating uncertainty — matches near these are rejected
_HEDGE_WORDS = [
    "partial", "resembles", "similar to", "like a", "not a", "no ",
    "sort of", "somewhat", "almost", "appears to be", "reminiscent",
    "might be", "could be", "possibly", "perhaps", "suggests a",
    "suggesting", "hint of", "slight", "vague",
]


def _is_hedged(text: str, match_start: int, window: int = 45) -> bool:
    """True if the keyword match is surrounded by hedge/uncertainty language."""
    start   = max(0, match_start - window)
    context = text[start: match_start + window]
    return any(h in context for h in _HEDGE_WORDS)


def grep_symbols(text: str) -> list[dict]:
    """Match text against SYMBOL_KEYWORDS, rejecting hedged matches.

    Returns list of {key, confidence, bbox} dicts.
    Confidence is 80 + 5 per additional trigger match, capped at 95.
    """
    combined   = text.lower()
    detections = []
    for key, triggers in SYMBOL_KEYWORDS.items():
        matched = []
        for t in triggers:
            pos = combined.find(t)
            if pos == -1:
                continue
            if _is_hedged(combined, pos):
                continue
            matched.append(t)
        if matched:
            conf = min(95, 80 + len(matched) * 5)
            detections.append({"key": key, "confidence": conf,
                                "bbox": [0.0, 0.0, 1.0, 1.0]})
    return detections


# ── Ollama helpers ────────────────────────────────────────────────────────────

def check_ollama(model: str) -> None:
    """Preflight — verify Ollama is running and model is pulled."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE}.\n"
            f"Start it with:  ollama serve\nError: {e}"
        )
    available = [m["name"] for m in data.get("models", [])]
    matched   = [n for n in available if n == model or n.startswith(model.split(":")[0])]
    if not matched:
        raise RuntimeError(
            f"Model '{model}' is not pulled.\n"
            f"Run:  ollama pull {model}\n"
            f"Available: {available or '(none)'}"
        )
    print(f"  Ollama OK — using model: {matched[0]}")


def call_ollama(img_b64: str, prompt: str, model: str) -> str:
    """POST to Ollama /api/chat with vision content. Returns raw response text."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "model":    model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream":   False,
        "options":  {"temperature": 0, "seed": 42, "num_predict": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL}.\nStart: ollama serve\n{e}")


def parse_json(raw: str) -> dict:
    """Extract and parse a JSON object from VLM response."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_mark_image(stem: str, ras_dir: Path, seg_dir: Path) -> Path | None:
    """Return path to the best available tight-crop image for this mark."""
    for candidate in [
        ras_dir / f"{stem}_isolated_crop.png",
        seg_dir / f"{stem}_isolated.png",
    ]:
        if candidate.exists():
            return candidate
    return None


def image_to_base64(path: Path) -> str:
    """Load image, composite onto white if RGBA, return base64 string."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_mask_array(stem: str, ras_dir: Path) -> "np.ndarray | None":
    """Load binary mask crop. Returns H×W bool array or None."""
    candidate = ras_dir / f"{stem}_mask_crop.png"
    if not candidate.exists():
        candidate = ras_dir / f"{stem}_isolated_crop.png"
        if not candidate.exists():
            return None
        img = Image.open(candidate)
        if img.mode == "RGBA":
            return np.array(img.split()[3]) > 128
        return None
    img = Image.open(candidate).convert("L")
    return np.array(img) > 128


def collect_stems(ras_dir: Path, seg_dir: Path) -> list[str]:
    """Find all processable mark stems across rasters and segmented dirs."""
    stems = set()
    if ras_dir.exists():
        stems |= {p.stem.replace("_isolated_crop", "")
                  for p in ras_dir.glob("*_isolated_crop.png")}
    if seg_dir.exists():
        stems |= {p.stem.replace("_isolated", "")
                  for p in seg_dir.glob("*_isolated.png")}
    return sorted(stems)


# ── Rendering helpers ─────────────────────────────────────────────────────────

DOT_COLOURS = [
    (230, 25, 75),  (60, 180, 75),   (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180),  (70, 240, 240), (240, 50, 230),
    (188, 246, 12), (250, 190, 212), (0, 128, 128),  (220, 190, 255),
]


def word_wrap(text: str, width: int) -> list[str]:
    """Wrap text to lines of at most `width` characters."""
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            if line:
                lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return lines


def draw_pixel_overlay(canvas: "Image.Image", mask_arr: "np.ndarray",
                       bbox_norm: list, colour: tuple, alpha: float = 0.45) -> None:
    """Colour all mark pixels within the normalised bbox on the canvas (in-place)."""
    cw, ch = canvas.size
    x1 = int(bbox_norm[0] * cw);  y1 = int(bbox_norm[1] * ch)
    x2 = int(bbox_norm[2] * cw);  y2 = int(bbox_norm[3] * ch)
    x1, x2 = max(0, min(x1, x2)), min(cw - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(ch - 1, max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    mh, mw = mask_arr.shape
    if (mw, mh) != (cw, ch):
        mask_img     = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(
            (cw, ch), Image.NEAREST)
        mask_resized = np.array(mask_img) > 128
    else:
        mask_resized = mask_arr

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ov_arr  = np.array(overlay)
    region  = mask_resized[y1:y2+1, x1:x2+1]
    ov_arr[y1:y2+1, x1:x2+1, 0][region] = colour[0]
    ov_arr[y1:y2+1, x1:x2+1, 1][region] = colour[1]
    ov_arr[y1:y2+1, x1:x2+1, 2][region] = colour[2]
    ov_arr[y1:y2+1, x1:x2+1, 3][region] = int(255 * alpha)
    overlay = Image.fromarray(ov_arr, "RGBA")

    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.paste(overlay, mask=overlay)
    draw = ImageDraw.Draw(canvas_rgba)
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
    canvas.paste(canvas_rgba.convert("RGB"))
