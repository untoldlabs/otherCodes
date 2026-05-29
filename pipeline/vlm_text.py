"""
vlm_text.py — Text / letter-focused VLM detection.

PROMPT PHILOSOPHY — TEXT / LETTERS / NUMBERS
=============================================
Focused specifically on textual content: individual letters, numbers,
words, tags, script style, and any letter-like or number-like glyphs.
Graffiti tags are fundamentally text-based — this pass ensures thorough
letter extraction even when letter forms are highly stylised.

  Pass 1  : Re-examines the image focused ONLY on text, letters, numbers.
  Pass 2  : Re-examines image + Pass 1 context → certain / ambiguous.
             Certain = clearly readable characters.
             Ambiguous = stylised characters that are guessed.
  Grep    : Runs on certain and ambiguous text.
             → certain_{key} / ambiguous_{key} CSV columns.
             Note: most matches here will be letter-related symbols.

Outputs:
  data/vlm_scores_text.csv  — one row per mark
  data/vlm_text/            — annotated report card images

Usage:
    conda activate othercodes
    python3 pipeline/vlm_text.py --project /path/to/project
    python3 pipeline/vlm_text.py --project ~/proj --stems IMG_0001
"""

import argparse
import csv
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))

from pipeline.vlm_core import (
    SYMBOL_DISPLAY, SYMBOL_KEYS,
    DOT_COLOURS, CONFIDENCE_THRESHOLD,
    call_ollama, check_ollama, parse_json,
    load_mark_image, image_to_base64, load_mask_array, collect_stems,
    grep_symbols, word_wrap, draw_pixel_overlay,
    VLM_MODEL,
)
from PIL import Image, ImageDraw

RAS_DIR = SEG_DIR = OUT_DIR = CSV_OUT = None
VARIANT = "text"


def _init_paths(project_root: Path):
    global RAS_DIR, SEG_DIR, OUT_DIR, CSV_OUT
    RAS_DIR = project_root / "data" / "rasters"
    SEG_DIR = project_root / "data" / "segmented"
    OUT_DIR = project_root / "data" / f"vlm_{VARIANT}"
    CSV_OUT = project_root / "data" / f"vlm_scores_{VARIANT}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_pass1_prompt() -> str:
    """Pass 1 — focused look at TEXT, LETTERS, NUMBERS only."""
    return """You are analysing a segmented graffiti tag or handmade mark on a white background.

Look carefully at this image. Focus SPECIFICALLY on text, letters, and numbers:

  - Every individual letter you can identify (A-Z, a-z)
  - Every number you can identify (0-9)
  - Complete words or tags that letters spell out
  - Writing style: bubble letters, wildstyle, simple script, block letters, etc.
  - Script type: latin, arabic, chinese, cyrillic, other
  - Any letter-like or number-like shapes, even if stylised or distorted
  - Characters that might be from non-latin scripts

Try to read the tag. Graffiti writers often use stylised letterforms —
look for the underlying letter structure even in abstract-looking strokes.

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "description": "Description of text content found: letters, words, style. E.g. 'The letters R, U, L spelling RUL in bubble style'. Or 'no text detected'.",
  "letters": ["R", "U", "L"],
  "words": ["RUL"],
  "text_script": "latin",
  "style": "bubble letters"
}

Rules:
- "letters": list every readable individual character.
- "words": any complete words the letters spell out.
- "text_script": "latin", "arabic", "chinese", "cyrillic", "other", or "none".
- "style": brief description of the lettering style, or "none".
- Look at the whole image including corners and floating elements.
"""


def _build_pass2_prompt(p1_description: str, p1_letters: list, p1_words: list) -> str:
    letters_str = ", ".join(p1_letters) if p1_letters else "none"
    words_str   = ", ".join(p1_words)   if p1_words   else "none"
    return f"""You just examined this graffiti mark for text content and found:

Description: "{p1_description}"
Letters identified: {letters_str}
Words identified: {words_str}

Now look at the image again carefully. Decide what text you can be confident about
versus what is uncertain:

  CERTAIN  — letters/words you can clearly and confidently read
  AMBIGUOUS — characters that might be present but are unclear, stylised
               beyond readability, or could be read multiple ways

Respond with ONLY a JSON object:
{{
  "certain_letters": ["list", "of", "confident", "characters"],
  "certain_words": ["list", "of", "confident", "words"],
  "ambiguous_letters": ["possible", "characters"],
  "ambiguous_words": ["possible", "words"],
  "certain": "description of clearly readable text",
  "ambiguous": "description of uncertain/stylised text"
}}

Leave lists as [] and strings as "" if nothing belongs there.
"""


def render_report(stem: str, img_path: Path,
                  p1_desc: str, certain_text: str, ambiguous_text: str,
                  certain_letters: list, ambiguous_letters: list,
                  certain_words: list, ambiguous_words: list,
                  text_style: str, text_script: str) -> None:
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

    panel_w = 460
    canvas  = Image.new("RGB", (display_w + panel_w, display_h), (30, 30, 35))
    canvas.paste(mark_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = display_w + 14
    y  = 12

    def txt(text, colour=(220, 220, 220)):
        nonlocal y
        draw.text((x0, y), text, fill=colour)
        y += 18

    def gap(px=6):
        nonlocal y
        y += px

    txt(f"{stem}  [text]", colour=(255, 255, 100))
    gap()
    txt("PASS 1 — TEXT SCAN:", colour=(160, 160, 255))
    for line in word_wrap(p1_desc or "(no description)", 48):
        txt(line, colour=(190, 190, 190))
    gap()
    txt("CERTAIN TEXT:", colour=(80, 220, 80))
    if certain_words:
        txt(f"  Words:   {' | '.join(certain_words)}", colour=(150, 240, 150))
    if certain_letters:
        txt(f"  Letters: {'  '.join(certain_letters)}", colour=(150, 240, 150))
    if certain_text:
        for line in word_wrap(certain_text, 48):
            txt(f"  {line}", colour=(160, 220, 160))
    if not certain_words and not certain_letters and not certain_text:
        txt("  (none)", colour=(150, 150, 150))
    gap()
    txt("AMBIGUOUS TEXT:", colour=(220, 180, 60))
    if ambiguous_words:
        txt(f"  Words:   {' | '.join(ambiguous_words)}", colour=(220, 200, 120))
    if ambiguous_letters:
        txt(f"  Letters: {'  '.join(ambiguous_letters)}", colour=(220, 200, 120))
    if ambiguous_text:
        for line in word_wrap(ambiguous_text, 48):
            txt(f"  {line}", colour=(200, 170, 100))
    if not ambiguous_words and not ambiguous_letters and not ambiguous_text:
        txt("  (none)", colour=(150, 150, 150))
    gap()
    if text_style and text_style not in ("none", ""):
        txt(f"STYLE: {text_style}", colour=(180, 180, 255))
    if text_script and text_script != "none":
        txt(f"SCRIPT: {text_script}", colour=(180, 180, 255))

    canvas.save(OUT_DIR / f"{stem}_report.png")


def process_mark(stem: str, model: str) -> dict | None:
    img_path = load_mark_image(stem, RAS_DIR, SEG_DIR)
    if img_path is None:
        print(f"  ✗ No image for {stem}")
        return None

    t0 = time.time()
    try:
        img_b64 = image_to_base64(img_path)

        print(f"  {stem}  — pass 1 (text)…")
        raw1    = call_ollama(img_b64, _build_pass1_prompt(), model)
        p1      = parse_json(raw1)
        p1_desc = p1.get("description", "")
        letters = p1.get("letters", []) or []
        words   = p1.get("words",   []) or []
        script  = p1.get("text_script", "none") or "none"
        style   = p1.get("style", "none") or "none"

        print(f"  {stem}  — pass 2 (consolidate)…")
        raw2   = call_ollama(img_b64, _build_pass2_prompt(p1_desc, letters, words), model)
        p2     = parse_json(raw2)

        certain_letters  = p2.get("certain_letters",  []) or []
        certain_words    = p2.get("certain_words",    []) or []
        ambiguous_letters = p2.get("ambiguous_letters", []) or []
        ambiguous_words  = p2.get("ambiguous_words",  []) or []
        certain_text     = p2.get("certain",   "") or ""
        ambiguous_text   = p2.get("ambiguous", "") or ""
        if isinstance(certain_text,  list): certain_text  = ", ".join(certain_text)
        if isinstance(ambiguous_text, list): ambiguous_text = ", ".join(ambiguous_text)

        print(f"    → certain letters:  {certain_letters}")
        print(f"    → ambiguous letters:{ambiguous_letters}")

        # Grep on text description for any symbol keywords (uncommon but possible)
        certain_dets   = grep_symbols(certain_text)
        ambiguous_dets = grep_symbols(ambiguous_text)

    except Exception as e:
        print(f"  ✗ {stem}: {e}")
        return None

    elapsed = time.time() - t0
    print(f"  {stem}  ({elapsed:.1f}s)")

    render_report(stem, img_path, p1_desc, certain_text, ambiguous_text,
                  certain_letters, ambiguous_letters,
                  certain_words, ambiguous_words, style, script)

    certain_map   = {d["key"]: d["confidence"] for d in certain_dets}
    ambiguous_map = {d["key"]: d["confidence"] for d in ambiguous_dets}

    summary = {
        "stem":              stem,
        "p1_description":    p1_desc,
        "certain_text":      certain_text,
        "ambiguous_text":    ambiguous_text,
        "certain_letters":   " ".join(str(l) for l in certain_letters),
        "certain_words":     " | ".join(str(w) for w in certain_words),
        "ambiguous_letters": " ".join(str(l) for l in ambiguous_letters),
        "ambiguous_words":   " | ".join(str(w) for w in ambiguous_words),
        "text_script":       script,
        "text_style":        style,
        "n_certain":         len(certain_dets),
        "n_ambiguous":       len(ambiguous_dets),
        "has_certain_text":  int(bool(certain_words or certain_letters)),
    }
    for key in SYMBOL_KEYS:
        summary[f"certain_{key}"]   = certain_map.get(key, 0)
        summary[f"ambiguous_{key}"] = ambiguous_map.get(key, 0)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=f"VLM {VARIANT}-focused detection: certain + ambiguous tiers")
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--model", default=VLM_MODEL)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve() if args.project \
                   else Path(__file__).parent.parent
    _init_paths(project_root)

    print(f"\nProject  : {project_root}")
    print(f"Variant  : {VARIANT}")
    print(f"Model    : {args.model}")

    try:
        check_ollama(args.model)
    except RuntimeError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    print()

    stems = collect_stems(RAS_DIR, SEG_DIR)
    if args.stems:
        stems = [s for s in stems if s in args.stems]
    if args.skip_existing:
        stems = [s for s in stems if not (OUT_DIR / f"{s}_report.png").exists()]
    if not stems:
        print("No marks to process.")
        return

    print(f"Processing {len(stems)} marks...\n")

    _sum_cols = (["stem", "p1_description", "certain_text", "ambiguous_text",
                  "certain_letters", "certain_words", "ambiguous_letters", "ambiguous_words",
                  "text_script", "text_style", "n_certain", "n_ambiguous", "has_certain_text"]
                 + [f"certain_{k}"  for k in SYMBOL_KEYS]
                 + [f"ambiguous_{k}" for k in SYMBOL_KEYS])

    csv_new = not CSV_OUT.exists()
    csv_f   = open(CSV_OUT, "a", newline="")
    csv_w   = csv.DictWriter(csv_f, fieldnames=_sum_cols)
    if csv_new:
        csv_w.writeheader()

    n_done = 0
    try:
        for stem in stems:
            result = process_mark(stem, args.model)
            if result is None:
                continue
            csv_w.writerow(result)
            csv_f.flush()
            n_done += 1
    finally:
        csv_f.close()

    print(f"\n✓ CSV → {CSV_OUT}")
    print(f"✓ Reports → {OUT_DIR}/")
    print(f"  {n_done} marks processed\n")


if __name__ == "__main__":
    main()
