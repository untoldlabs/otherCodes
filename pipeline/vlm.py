"""
vlm.py — Symbol detection using Qwen2.5-VL via Ollama.

Uses a local Vision Language Model (Qwen2.5-VL:7B) running via Ollama to
identify symbols, letters, and words in each segmented mark.

Temperature=0 ensures fully deterministic output:
  same image + same prompt + same model weights = identical result every run.

Architecture: vision encoder slices the image into patches → large language
model reasons over patch embeddings + prompt → structured JSON output.
This is the same architecture as Claude, GPT-4V, and other frontier VLMs,
running locally and free.

Outputs:
  data/vlm_scores.csv  — one row per mark (detected symbols + OCR text)
  data/vlm/            — annotated report card image per mark

Prerequisites:
  brew install ollama
  ollama pull qwen2.5vl:7b
  ollama serve          (in a separate terminal, or it auto-starts on Mac)

Usage:
    conda activate othercodes
    python3 pipeline/vlm.py --project /path/to/project
    python3 pipeline/vlm.py --project ~/proj --stems IMG_0001 IMG_0002
    python3 pipeline/vlm.py --model qwen2.5vl:7b  # default model
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE          = "http://localhost:11434"
OLLAMA_URL           = f"{OLLAMA_BASE}/api/chat"   # /api/chat handles vision; /api/generate 404s on some models
VLM_MODEL            = "qwen2.5vl:7b"
CONFIDENCE_THRESHOLD = 76   # kept for display only; grep scoring starts at 80

# ── Paths ─────────────────────────────────────────────────────────────────────

RAS_DIR = SEG_DIR = VLM_DIR = CSV_VLM = None

def _init_paths(project_root: Path, run_id: int | None = None):
    global RAS_DIR, SEG_DIR, VLM_DIR, CSV_VLM
    RAS_DIR  = project_root / "data" / "rasters"
    SEG_DIR  = project_root / "data" / "segmented"
    suffix   = f"_run{run_id}" if run_id is not None else ""
    VLM_DIR  = project_root / "data" / f"vlm{suffix}"
    CSV_VLM  = project_root / "data" / f"vlm_scores{suffix}.csv"
    VLM_DIR.mkdir(parents=True, exist_ok=True)


# ── Symbol list ───────────────────────────────────────────────────────────────
# Keys must be stable — they become CSV column names.
# Display names are shown in the prompt and report card.

SYMBOLS = [
    # Geometric primitives
    ("accent_dot",    "Floating accent dot / isolated point (small mark floating near or above the tag, a decorative motif)"),
    ("arch",          "Arch / rainbow arc"),
    ("zigzag",        "Zig-zag / lightning bolt"),
    ("circle",        "Circle"),
    ("oval",          "Oval / ellipse"),
    ("triangle",      "Triangle"),
    ("square",        "Square / rectangle"),
    ("crosshatch",    "Crosshatch / grid"),
    # Symbolic shapes
    ("cruciform",     "Cross / plus sign"),
    ("open_angle",    "V-shape / chevron"),
    ("asterisk",      "Asterisk / starburst (*)"),
    ("star",          "Five-pointed star"),
    ("spiral",        "Spiral / swirl"),
    ("meander",       "Meander / Greek key"),
    ("branching",     "Branching / tree"),
    ("cordiform",     "Heart / cordiform"),
    ("arrow",         "Arrow (with shaft)"),
    ("arrowhead",     "Arrowhead only (no shaft)"),
    ("wheel",         "Sun symbol / wheel / concentric circles"),
    ("serpentiform",  "Snake / S-curve / serpentiform"),
    ("eye",           "Eye symbol (open or closed, realistic or schematic — single eye)"),
    ("peace",         "Peace symbol (circle with a central vertical line and two downward diagonal lines, exactly like ☮)"),
    ("drip",          "Paint drip / teardrop (a drop hanging off a stroke, common graffiti motif)"),
    # Figurative
    ("face",          "Face (any — smiley, schematic, mask)"),
    ("hand_negative", "Negative hand stencil"),
    ("hand_positive", "Positive handprint"),
    ("partial_hand",  "Partial hand / missing fingers"),
    # Rare / prehistoric
    ("claviform",     "Claviform / lollipop shape"),
    ("tectiform",     "Tectiform / house outline"),
    ("penniform",     "Penniform / feather"),
    ("scalariform",   "Scalariform / ladder"),
    ("flabelliform",  "Flabelliform / fan shape"),
    ("reniform",      "Reniform / kidney shape"),
    ("trilobite",     "Trilobite / segmented oval"),
    ("phytomorph",    "Phytomorph / leaf or fern"),
    ("cupule",        "Cupule / carved depression"),
    ("finger_fluting","Finger fluting / parallel wavy lines"),
    ("thumb",         "Thumb stencil"),
    ("geometric",     "Complex geometric / compound symbol"),
]

SYMBOL_KEYS    = [s[0] for s in SYMBOLS]
SYMBOL_DISPLAY = {s[0]: s[1] for s in SYMBOLS}

# Approximate location grid for annotating the report image
_LOC_MAP = {
    "top-left":      (0.20, 0.20), "top-centre":    (0.50, 0.15),
    "top-right":     (0.80, 0.20), "centre-left":   (0.15, 0.50),
    "centre":        (0.50, 0.50), "centre-right":  (0.85, 0.50),
    "bottom-left":   (0.20, 0.80), "bottom-centre": (0.50, 0.85),
    "bottom-right":  (0.80, 0.80),
}

# Distinct colours for location dots (one per detected symbol, cycling)
_DOT_COLOURS = [
    (230, 25, 75),  (60, 180, 75),  (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240),  (240, 50, 230),
    (188, 246, 12), (250, 190, 212),(0, 128, 128),   (220, 190, 255),
]


# ── Config ────────────────────────────────────────────────────────────────────

# ── Prompts ───────────────────────────────────────────────────────────────────

def _build_description_prompt() -> str:
    """Pass 1 — free description, no symbol list shown. Model writes what it sees."""
    return """You are analysing a segmented graffiti tag or handmade mark on a white background.

Describe what you see in detail. Be specific about every shape and element.
Do NOT use vague terms like "abstract shapes", "decorative elements", or "various forms" —
instead name the actual shapes: circle, spiral, loop, heart, oval, face, arrow, etc.

Respond with ONLY a JSON object (no markdown, no explanation):

{
  "description": "Detailed description naming every specific shape and element you see.",
  "letters": [],
  "words": [],
  "text_script": "none"
}

Rules:
- "letters": every readable character you can identify (e.g. ["R", "U", "L"]).
- "words": any complete readable words (e.g. ["RUL", "KINGS"]).
- "text_script": one of "latin", "arabic", "chinese", "cyrillic", "other", "none".
- If you see letters forming a tag, list them.
- Look at the whole image — top, bottom, corners, floating elements near the tag.
"""


def _build_figures_prompt(description: str) -> str:
    """Pass 2a — focused on figurative content only.
    Faces, characters, and body parts are consistently under-reported in a
    generic pass, so we ask about them exclusively here.
    """
    return f"""You described this graffiti image as: "{description}"

Look ONLY for figurative content — any human, animal, or character elements.
Be specific: name every face, head, eye, mouth, nose, skull, body, limb, hand,
figure, character, portrait, smiley, or creature you can identify.
Include schematic or abstract versions — a simple circle with two dots for eyes counts as a face.
Include partial features — if you see just eyes, or just a mouth, name them.
If there is nothing figurative at all, say "none".

Do NOT copy the example — write your own from the description above.

Respond with ONLY a JSON object:
{{
  "figures": "square jaw, two small dots"
}}
"""


def _build_geometry_prompt(description: str) -> str:
    """Pass 2b — focused on geometric shapes only."""
    return f"""You described this graffiti image as: "{description}"

Look ONLY for geometric shapes — pure forms with no symbolic meaning.
Name every: circle, oval, loop, arch, arc, zigzag, triangle, square, rectangle,
crosshatch, grid, dot, line, curve, spiral, meander, branching, or ladder shape.
Include small or decorative versions — a tiny floating dot counts.
If there are no geometric shapes, say "none".

Do NOT copy the example — write your own from the description above.

Respond with ONLY a JSON object:
{{
  "geometry": "large arch, two dots, spiral"
}}
"""


def _build_motifs_prompt(description: str) -> str:
    """Pass 2c — focused on symbolic motifs and cultural glyphs only."""
    return f"""You described this graffiti image as: "{description}"

Look ONLY for symbolic motifs and cultural glyphs — shapes that carry meaning.
Name every: heart, arrow, star, cross, peace sign, drip, teardrop, eye symbol,
crown, sun, wheel, snake, feather, leaf, hand stencil, or any other recognisable symbol or emblem.
If there are no symbols or motifs, say "none".

Do NOT copy the example — write your own from the description above.

Respond with ONLY a JSON object:
{{
  "motifs": "drip, arrow"
}}
"""


def _extract_pass2_text(raw: str, key: str) -> str:
    """Parse a focused pass response, returning a plain string."""
    parsed = _parse_json(raw)
    val = parsed.get(key, "")
    if isinstance(val, list):
        val = ", ".join(str(v) for v in val)
    val = str(val).strip()
    return "" if val.lower() in ("none", "n/a", "") else val


# ── Symbol keyword grep ────────────────────────────────────────────────────────
# For each symbol key, list of text triggers that indicate it's present.
# Matched against Pass 2 symbol distillation text.

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


# Words that, when appearing near a keyword match, indicate the model is hedging
# rather than asserting the symbol is present.
_HEDGE_WORDS = [
    "partial", "resembles", "similar to", "like a", "not a", "no ",
    "sort of", "somewhat", "almost", "appears to be", "reminiscent",
    "might be", "could be", "possibly", "perhaps", "suggests a",
    "suggesting", "hint of", "slight", "vague",
]

def _is_hedged(text: str, match_start: int, window: int = 45) -> bool:
    """Return True if the keyword match is surrounded by hedge/uncertainty language."""
    start   = max(0, match_start - window)
    context = text[start: match_start + window]
    return any(h in context for h in _HEDGE_WORDS)


def _grep_symbols(symbols_text: str) -> list[dict]:
    """Match Pass 2 symbol distillation text against SYMBOL_KEYWORDS.

    A keyword match is accepted only if it is NOT surrounded by hedging language
    (e.g. "resembles a circle", "partial circle" are rejected).
    Confidence is boosted by the number of unambiguous trigger matches.
    """
    combined = symbols_text.lower()
    detections = []
    for key, triggers in SYMBOL_KEYWORDS.items():
        matched = []
        for t in triggers:
            pos = combined.find(t)
            if pos == -1:
                continue
            if _is_hedged(combined, pos):
                continue          # keyword present but hedged — skip
            matched.append(t)
        if matched:
            conf = min(95, 80 + len(matched) * 5)
            detections.append({"key": key, "confidence": conf,
                               "bbox": [0.0, 0.0, 1.0, 1.0]})
    return detections


def _build_grounding_prompt(symbol_display: str) -> str:
    """Pass 2 — native Qwen2.5-VL grounding prompt for a single symbol.
    The model responds with <|box_start|>(x1,y1),(x2,y2)<|box_end|> tokens
    which are tied to actual patch positions, not guessed coordinates.
    """
    return f"Detect {symbol_display} in the image."


def _parse_grounding_tokens(raw: str) -> list[float] | None:
    """Parse Qwen2.5-VL native grounding tokens from response text.

    Qwen outputs: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    Coordinates are on a 0–999 scale (1000×1000 grid over the image).
    Returns first bbox as normalised [x1,y1,x2,y2] or None if not found.
    """
    pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern, raw)
    if not matches:
        return None
    x1, y1, x2, y2 = (int(v) for v in matches[0])
    # Clamp + normalise from 0-999 → 0.0-1.0
    return [
        max(0.0, min(1.0, x1 / 999)),
        max(0.0, min(1.0, y1 / 999)),
        max(0.0, min(1.0, x2 / 999)),
        max(0.0, min(1.0, y2 / 999)),
    ]


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_mark_image(stem: str) -> Path | None:
    """Return path to the best available tight-crop image for this mark."""
    for candidate in [
        RAS_DIR / f"{stem}_isolated_crop.png",
        SEG_DIR / f"{stem}_isolated.png",
    ]:
        if candidate.exists():
            return candidate
    return None


def _image_to_base64(path: Path) -> str:
    """Load image, composite onto white if RGBA, return base64 string."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Ollama call ───────────────────────────────────────────────────────────────

def _check_ollama(model: str) -> None:
    """Preflight: verify Ollama is running and the model is pulled. Raises with clear message."""
    import urllib.request, urllib.error
    tags_url = f"{OLLAMA_BASE}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE}.\n"
            f"Start it with:  ollama serve\n"
            f"Error: {e}"
        )

    available = [m["name"] for m in data.get("models", [])]
    # Exact match or prefix match (e.g. "qwen2.5vl:7b" matches "qwen2.5vl:7b-q4_K_M")
    matched = [n for n in available if n == model or n.startswith(model.split(":")[0])]
    if not matched:
        raise RuntimeError(
            f"Model '{model}' is not pulled.\n"
            f"Run:  ollama pull {model}\n"
            f"Available models: {available or '(none)'}"
        )
    print(f"  Ollama OK — using model: {matched[0]}")


def _call_ollama(img_b64: str, prompt: str, model: str) -> str:
    """POST to Ollama /api/chat with vision content. Returns raw response text."""
    import urllib.request, urllib.error

    # /api/chat format — required for vision models in Ollama
    payload = json.dumps({
        "model":  model,
        "messages": [
            {
                "role":    "user",
                "content": prompt,
                "images":  [img_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,      # fully deterministic
            "seed": 42,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            return result.get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {e.code} at {OLLAMA_URL}.\n"
            f"Response: {body}\n"
            f"Make sure the model is pulled: ollama pull {model}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}.\n"
            f"Start it with: ollama serve\n"
            f"Error: {e}"
        )


def _parse_json(raw: str) -> dict:
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


def _parse_json_array(raw: str) -> list:
    """Extract and parse a JSON array from VLM response."""
    try:
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def _build_result(parsed_desc: dict, further_detail: str, detections: list[dict]) -> dict:
    """Assemble final result dict from description pass + grep detections."""
    return {
        "description":    parsed_desc.get("description", ""),
        "further_detail": further_detail,
        "detections":     detections,
        "letters":        parsed_desc.get("letters", []),
        "words":          parsed_desc.get("words", []),
        "text_script":    parsed_desc.get("text_script", "none"),
    }


def _apply_localisations(result: dict, img_b64: str, model: str) -> dict:
    """Pass 2: one native Qwen grounding call per detection — real patch-level bboxes."""
    for det in result["detections"]:
        display = SYMBOL_DISPLAY[det["key"]]
        prompt  = _build_grounding_prompt(display)
        try:
            raw  = _call_ollama(img_b64, prompt, model)
            bbox = _parse_grounding_tokens(raw)
            if bbox:
                det["bbox"] = bbox
                print(f"      ↳ bbox {det['key']}: {[round(v,3) for v in bbox]}")
            else:
                print(f"      ↳ bbox {det['key']}: no grounding tokens in response")
        except Exception as e:
            print(f"      ↳ bbox {det['key']}: error — {e}")
    return result


# ── Visualisation ─────────────────────────────────────────────────────────────

def _load_mask_array(stem: str) -> "np.ndarray | None":
    """Load binary mask crop (white mark on black). Returns H×W bool array or None."""
    candidate = RAS_DIR / f"{stem}_mask_crop.png"
    if not candidate.exists():
        # Fall back: derive from isolated crop alpha channel
        candidate = RAS_DIR / f"{stem}_isolated_crop.png"
        if not candidate.exists():
            return None
        img = Image.open(candidate)
        if img.mode == "RGBA":
            return np.array(img.split()[3]) > 128
        return None
    img = Image.open(candidate).convert("L")
    return np.array(img) > 128


def _draw_pixel_overlay(canvas: "Image.Image", mask_arr: "np.ndarray",
                        bbox_norm: list, colour: tuple, alpha: float = 0.45) -> None:
    """Colour all mark pixels within the normalised bbox on the canvas (in-place)."""
    cw, ch = canvas.size
    x1 = int(bbox_norm[0] * cw)
    y1 = int(bbox_norm[1] * ch)
    x2 = int(bbox_norm[2] * cw)
    y2 = int(bbox_norm[3] * ch)
    x1, x2 = max(0, min(x1, x2)), min(cw - 1, max(x1, x2))  # ensure x1 <= x2
    y1, y2 = max(0, min(y1, y2)), min(ch - 1, max(y1, y2))   # ensure y1 <= y2
    if x2 - x1 < 2 or y2 - y1 < 2:
        return  # degenerate bbox — skip

    # Resize mask to canvas dims if needed
    mh, mw = mask_arr.shape
    if (mw, mh) != (cw, ch):
        mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(
            (cw, ch), Image.NEAREST)
        mask_resized = np.array(mask_img) > 128
    else:
        mask_resized = mask_arr

    # Build coloured overlay layer
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ov_arr  = np.array(overlay)
    region  = mask_resized[y1:y2+1, x1:x2+1]
    ov_arr[y1:y2+1, x1:x2+1, 0][region] = colour[0]
    ov_arr[y1:y2+1, x1:x2+1, 1][region] = colour[1]
    ov_arr[y1:y2+1, x1:x2+1, 2][region] = colour[2]
    ov_arr[y1:y2+1, x1:x2+1, 3][region] = int(255 * alpha)
    overlay = Image.fromarray(ov_arr, "RGBA")

    # Also draw bbox rectangle outline
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.paste(overlay, mask=overlay)
    draw = ImageDraw.Draw(canvas_rgba)
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
    canvas.paste(canvas_rgba.convert("RGB"))


def _word_wrap(text: str, width: int) -> list[str]:
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


def render_report(stem: str, img_path: Path, result: dict) -> None:
    """
    Save an annotated report card:
      Left:  mark with per-symbol pixel-level coloured overlays + bbox outlines.
      Right: symbol list with confidence %, letters/words, description.
    """
    # ── Load mark ──────────────────────────────────────────────────────────────
    mark_img = Image.open(img_path)
    if mark_img.mode == "RGBA":
        bg = Image.new("RGBA", mark_img.size, (255, 255, 255, 255))
        bg.paste(mark_img, mask=mark_img.split()[3])
        mark_img = bg.convert("RGB")
    else:
        mark_img = mark_img.convert("RGB")

    display_h = 600
    aspect    = mark_img.width / mark_img.height
    display_w = max(200, int(display_h * aspect))
    mark_img  = mark_img.resize((display_w, display_h), Image.LANCZOS)

    # Try to load binary mask for pixel-level overlay
    mask_arr = _load_mask_array(stem)

    # ── Canvas ─────────────────────────────────────────────────────────────────
    panel_w  = 440
    canvas_w = display_w + panel_w
    canvas_h = display_h
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 35))
    canvas.paste(mark_img, (0, 0))

    # ── Pixel overlays ─────────────────────────────────────────────────────────
    detections = result.get("detections", [])
    mark_region = canvas.crop((0, 0, display_w, display_h))

    for i, det in enumerate(detections):
        colour = _DOT_COLOURS[i % len(_DOT_COLOURS)]
        bbox   = det["bbox"]
        if mask_arr is not None:
            _draw_pixel_overlay(mark_region, mask_arr, bbox, colour, alpha=0.50)
        else:
            # No mask — just draw the bbox rectangle
            draw_tmp = ImageDraw.Draw(mark_region)
            x1 = int(bbox[0] * display_w); y1 = int(bbox[1] * display_h)
            x2 = int(bbox[2] * display_w); y2 = int(bbox[3] * display_h)
            draw_tmp.rectangle([x1, y1, x2, y2], outline=colour, width=3)

    canvas.paste(mark_region, (0, 0))

    # ── Label dots + key on mark ───────────────────────────────────────────────
    draw = ImageDraw.Draw(canvas)
    for i, det in enumerate(detections):
        colour = _DOT_COLOURS[i % len(_DOT_COLOURS)]
        bbox   = det["bbox"]
        cx = int(((bbox[0] + bbox[2]) / 2) * display_w)
        cy = int(((bbox[1] + bbox[3]) / 2) * display_h)
        r  = 7
        draw.ellipse([cx-r-1, cy-r-1, cx+r+1, cy+r+1], fill=(0, 0, 0))
        draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r  ], fill=colour)
        label = f"{i+1}"
        draw.text((cx - 4, cy - 6), label, fill=(0, 0, 0))

    # ── Right panel ────────────────────────────────────────────────────────────
    x0 = display_w + 14
    y  = 12

    def txt(text, colour=(220, 220, 220)):
        nonlocal y
        draw.text((x0, y), text, fill=colour)
        y += 18

    def gap(px=6):
        nonlocal y
        y += px

    txt(stem, colour=(255, 255, 100))
    gap()

    txt("DESCRIPTION:", colour=(160, 160, 255))
    for line in _word_wrap(result.get("description", ""), 46):
        txt(line, colour=(190, 190, 190))

    further = result.get("further_detail", "")
    if further:
        gap(4)
        txt("SYMBOLS IDENTIFIED:", colour=(200, 160, 80))
        for line in _word_wrap(further, 46):
            txt(line, colour=(210, 185, 140))
    gap()

    txt("SYMBOLS DETECTED:", colour=(120, 200, 120))
    if detections:
        for i, det in enumerate(detections):
            colour = _DOT_COLOURS[i % len(_DOT_COLOURS)]
            conf   = det["confidence"]
            label  = f"  {i+1}. {SYMBOL_DISPLAY[det['key']]}  {conf}%"
            txt(label, colour=colour)
    else:
        txt("  (none above threshold)", colour=(150, 150, 150))
    gap()

    txt("TEXT:", colour=(120, 180, 255))
    words   = result.get("words", [])
    letters = result.get("letters", [])
    script  = result.get("text_script", "none")
    if words:
        txt(f"  Words:   {' | '.join(words)}", colour=(180, 220, 255))
    if letters:
        txt(f"  Letters: {'  '.join(letters)}", colour=(180, 220, 255))
    if script and script != "none":
        txt(f"  Script:  {script}", colour=(180, 220, 255))
    if not words and not letters:
        txt("  (no text detected)", colour=(150, 150, 150))

    gap()
    txt(f"Threshold: {CONFIDENCE_THRESHOLD}%", colour=(100, 100, 100))

    canvas.save(VLM_DIR / f"{stem}_report.png")


def render_annotated(stem: str, img_path: Path, result: dict) -> None:
    """
    Save a clean annotated image — mark only, with coloured bbox overlays and
    numbered dots. No side panel. Saved to data/vlm/{stem}_annotated.png.
    """
    mark_img = Image.open(img_path)
    if mark_img.mode == "RGBA":
        bg = Image.new("RGBA", mark_img.size, (255, 255, 255, 255))
        bg.paste(mark_img, mask=mark_img.split()[3])
        mark_img = bg.convert("RGB")
    else:
        mark_img = mark_img.convert("RGB")

    display_h = 600
    aspect    = mark_img.width / mark_img.height
    display_w = max(200, int(display_h * aspect))
    mark_img  = mark_img.resize((display_w, display_h), Image.LANCZOS)

    mask_arr  = _load_mask_array(stem)
    detections = result.get("detections", [])

    for i, det in enumerate(detections):
        colour = _DOT_COLOURS[i % len(_DOT_COLOURS)]
        if mask_arr is not None:
            _draw_pixel_overlay(mark_img, mask_arr, det["bbox"], colour, alpha=0.50)
        else:
            draw_tmp = ImageDraw.Draw(mark_img)
            x1 = int(det["bbox"][0] * display_w); y1 = int(det["bbox"][1] * display_h)
            x2 = int(det["bbox"][2] * display_w); y2 = int(det["bbox"][3] * display_h)
            draw_tmp.rectangle([x1, y1, x2, y2], outline=colour, width=3)

    draw = ImageDraw.Draw(mark_img)
    for i, det in enumerate(detections):
        colour = _DOT_COLOURS[i % len(_DOT_COLOURS)]
        bbox   = det["bbox"]
        cx = int(((bbox[0] + bbox[2]) / 2) * display_w)
        cy = int(((bbox[1] + bbox[3]) / 2) * display_h)
        r  = 7
        draw.ellipse([cx-r-1, cy-r-1, cx+r+1, cy+r+1], fill=(0, 0, 0))
        draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r  ], fill=colour)
        draw.text((cx - 4, cy - 6), str(i + 1), fill=(0, 0, 0))

    mark_img.save(VLM_DIR / f"{stem}_annotated.png")


# ── Process one mark ──────────────────────────────────────────────────────────

def process_mark(stem: str, model: str, detection_prompt: str) -> dict | None:
    img_path = _load_mark_image(stem)
    if img_path is None:
        print(f"  ✗ No image for {stem}")
        return None

    t0 = time.time()
    try:
        img_b64 = _image_to_base64(img_path)

        # ── Pass 1: free description — no symbol list shown ───────────────────
        raw1   = _call_ollama(img_b64, detection_prompt, model)
        parsed = _parse_json(raw1)
        desc_raw    = parsed.get("description", "")
        description = desc_raw if isinstance(desc_raw, str) else json.dumps(desc_raw)

        # ── Pass 2a: figures (faces, eyes, characters) ───────────────────────
        print(f"  {stem}  — pass 2a (figures)…")
        fig_text = _extract_pass2_text(
            _call_ollama(img_b64, _build_figures_prompt(description), model), "figures")
        print(f"    → figures:  {fig_text[:80] or '(none)'}")

        # ── Pass 2b: geometry (circles, arches, zigzags…) ────────────────────
        print(f"  {stem}  — pass 2b (geometry)…")
        geo_text = _extract_pass2_text(
            _call_ollama(img_b64, _build_geometry_prompt(description), model), "geometry")
        print(f"    → geometry: {geo_text[:80] or '(none)'}")

        # ── Pass 2c: motifs (hearts, arrows, drips, stars…) ──────────────────
        print(f"  {stem}  — pass 2c (motifs)…")
        mot_text = _extract_pass2_text(
            _call_ollama(img_b64, _build_motifs_prompt(description), model), "motifs")
        print(f"    → motifs:   {mot_text[:80] or '(none)'}")

        # Combine all three into one string for grep + display
        symbols_text = ", ".join(t for t in [fig_text, geo_text, mot_text] if t)

        # ── Grep: match all three pass outputs against keyword dict ──────────
        detections = _grep_symbols(symbols_text)
        print(f"    → grep matched: {[d['key'] for d in detections] or 'none'}")

        result = _build_result(parsed, symbols_text, detections)

        # ── Pass 2: native Qwen grounding for each matched symbol ─────────────
        if result["detections"]:
            result = _apply_localisations(result, img_b64, model)

    except Exception as e:
        print(f"  ✗ {stem}: {e}")
        return None

    elapsed = time.time() - t0

    render_report(stem, img_path, result)
    render_annotated(stem, img_path, result)

    detections = result.get("detections", [])
    words   = result.get("words", [])
    letters = result.get("letters", [])
    script  = result.get("text_script", "none")
    desc    = result.get("description", "")

    print(f"  {stem}  ({elapsed:.1f}s)")
    if detections:
        for d in detections:
            print(f"    {d['confidence']:3d}%  {SYMBOL_DISPLAY[d['key']]}")
    else:
        print(f"    Symbols : (none)")
    if words:
        print(f"    Words   : {words}")
    if letters:
        print(f"    Letters : {letters}")

    letters_str = " ".join(letters)
    words_str   = " | ".join(words)

    # ── Summary row (one per mark) for vlm_scores.csv ─────────────────────────
    conf_map = {}
    for d in detections:
        conf_map[d["key"]] = max(conf_map.get(d["key"], 0), d["confidence"])

    summary = {"stem": stem, "description": desc,
                "further_detail": result.get("further_detail", ""),
                "letters": letters_str, "words": words_str,
                "text_script": script,
                "n_symbols": len(set(d["key"] for d in detections)),
                "has_text": int(bool(words or letters))}
    for key in SYMBOL_KEYS:
        summary[f"sym_{key}"] = conf_map.get(key, 0)

    # ── Detection rows (one per detection) for vlm_detections.tsv ─────────────
    det_rows = []
    if detections:
        for i, d in enumerate(detections):
            bbox = d["bbox"]
            det_rows.append({
                "stem":            stem,
                "description":     desc,
                "letters":         letters_str,
                "words":           words_str,
                "text_script":     script,
                "threshold":       CONFIDENCE_THRESHOLD,
                "symbol_n":        i + 1,
                "symbol_key":      d["key"],
                "symbol_display":  SYMBOL_DISPLAY[d["key"]],
                "confidence":      d["confidence"],
                "bbox_x1":         round(bbox[0], 4),
                "bbox_y1":         round(bbox[1], 4),
                "bbox_x2":         round(bbox[2], 4),
                "bbox_y2":         round(bbox[3], 4),
            })
    else:
        # One row even for marks with no detections
        det_rows.append({
            "stem":           stem,
            "description":    desc,
            "letters":        letters_str,
            "words":          words_str,
            "text_script":    script,
            "threshold":      CONFIDENCE_THRESHOLD,
            "symbol_n":       0,
            "symbol_key":     "",
            "symbol_display": "",
            "confidence":     0,
            "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
        })

    return {"summary": summary, "detections": det_rows}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLM symbol detection via Qwen2.5-VL + Ollama")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project root (contains data/rasters/ or data/segmented/)")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Process only these stems (default: all)")
    parser.add_argument("--model", default=VLM_MODEL,
                        help=f"Ollama model name (default: {VLM_MODEL})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stems that already have a report image")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Replicate run number — saves to vlm_run{N}/ and vlm_scores_run{N}.csv")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve() if args.project \
                   else Path(__file__).parent.parent
    _init_paths(project_root, run_id=args.run_id)

    run_label = f" (run {args.run_id})" if args.run_id is not None else ""
    print(f"\nProject : {project_root}{run_label}")
    print(f"Model   : {args.model}")
    print(f"Ollama  : {OLLAMA_URL}")

    # Preflight — fail fast with a clear message
    try:
        _check_ollama(args.model)
    except RuntimeError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    print()

    # Collect stems — from rasters (post-analyze) or segmented (pre-analyze)
    stems = sorted({
        p.stem.replace("_isolated_crop", "")
        for p in RAS_DIR.glob("*_isolated_crop.png")
    } | {
        p.stem.replace("_isolated", "")
        for p in SEG_DIR.glob("*_isolated.png")
    })

    if args.stems:
        stems = [s for s in stems if s in args.stems]
    if args.skip_existing:
        stems = [s for s in stems
                 if not (VLM_DIR / f"{s}_report.png").exists()]

    if not stems:
        print("No marks to process.")
        return

    print(f"Processing {len(stems)} marks...\n")

    prompt   = _build_description_prompt()
    tsv_path = CSV_VLM.parent / "vlm_detections.tsv"

    # Build column lists from a dummy run so we know headers before any marks
    _sum_cols = (["stem", "description", "further_detail", "letters", "words",
                  "text_script", "n_symbols", "has_text"]
                 + [f"sym_{k}" for k in SYMBOL_KEYS])
    _det_cols = ["stem", "description", "letters", "words", "text_script",
                 "threshold", "symbol_n", "symbol_key", "symbol_display",
                 "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    # Open both files — append if they already exist (preserves prior runs)
    csv_new = not CSV_VLM.exists()
    tsv_new = not tsv_path.exists()

    csv_f = open(CSV_VLM,  "a", newline="")
    tsv_f = open(tsv_path, "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=_sum_cols)
    tsv_w = csv.DictWriter(tsv_f, fieldnames=_det_cols, delimiter="\t")
    if csv_new: csv_w.writeheader()
    if tsv_new: tsv_w.writeheader()

    n_done = n_symbols = n_text = n_detections = 0

    try:
        for stem in stems:
            result = process_mark(stem, args.model, prompt)
            if not result:
                continue
            csv_w.writerow(result["summary"])
            csv_f.flush()
            for row in result["detections"]:
                tsv_w.writerow(row)
            tsv_f.flush()
            n_done     += 1
            n_symbols  += result["summary"]["n_symbols"] > 0
            n_text     += result["summary"]["has_text"]
            n_detections += sum(1 for r in result["detections"] if r["symbol_key"])
    finally:
        csv_f.close()
        tsv_f.close()

    if not n_done:
        print("No results.")
        return

    print(f"\n✓ Summary CSV     → {CSV_VLM}")
    print(f"✓ Detections TSV  → {tsv_path}")
    print(f"  {n_done} marks processed")
    print(f"  {n_detections} symbol detections total")
    print(f"  {n_symbols} marks with symbols, {n_text} with text")
    print(f"  Report images    → {VLM_DIR}/\n")


if __name__ == "__main__":
    main()
