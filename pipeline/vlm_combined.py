"""
vlm_combined.py — Semi-focused, channelled VLM detection across all categories.

PROMPT PHILOSOPHY — COMBINED / CHANNELLED
==========================================
A single script that channels attention to ALL major categories in one sweep:
Figures, Geometry, Glyphs, and Text — without the depth of each dedicated
focused script, but with broader coverage in a single pass.

The key difference from vlm.py (neutral):
  - Pass 1 prompt explicitly names category channels, directing attention
    systematically across all four content types in sequence.
  - Pass 2 consolidates with certain / ambiguous — same as the focused scripts.

Use this as a middle ground:
  - More thorough than vlm.py neutral (explicit category attention)
  - Faster than running all four focused scripts individually
  - Useful as a quick sweep before deciding which focused script to run

  Pass 1  : Re-examines image with channelled attention across all 4 categories.
  Pass 2  : Re-examines image + Pass 1 context → certain / ambiguous.
  Grep    : Runs on certain and ambiguous text separately.
             → certain_{key} / ambiguous_{key} CSV columns.

Outputs:
  data/vlm_scores_combined.csv  — one row per mark
  data/vlm_combined/            — annotated report card images

Usage:
    conda activate othercodes
    python3 pipeline/vlm_combined.py --project /path/to/project
    python3 pipeline/vlm_combined.py --project ~/proj --stems IMG_0001
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
VARIANT = "combined"


def _init_paths(project_root: Path):
    global RAS_DIR, SEG_DIR, OUT_DIR, CSV_OUT
    RAS_DIR = project_root / "data" / "rasters"
    SEG_DIR = project_root / "data" / "segmented"
    OUT_DIR = project_root / "data" / f"vlm_{VARIANT}"
    CSV_OUT = project_root / "data" / f"vlm_scores_{VARIANT}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_pass1_prompt() -> str:
    """Pass 1 — channelled attention across all four content categories."""
    return """You are analysing a segmented graffiti tag or handmade mark on a white background.

Examine this image systematically across ALL four categories below.
For each category, list what you actually see — or write "none" if nothing is there.
Be specific: name actual elements, not vague descriptions.

1. FIGURES — faces, heads, skulls, masks, smiley faces, portraits, eyes+mouth combos,
   people, bodies, figures, hands, animals, cartoon characters, anthropomorphic forms:

2. GEOMETRY — circles, ovals, spirals, arches, arcs, zigzags, triangles, squares,
   crosses, chevrons, grids, wheels, meanders, branching, S-curves, loops, accent dots:

3. GLYPHS — hearts, arrows, stars, drips/teardrops, peace signs, sun symbols,
   feathers, fans, kidneys, prehistoric marks, any symbolic/iconic motif:

4. TEXT — letters (list each), numbers, words spelled out, writing style, script type:

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "figures":     "comma-separated list of figurative elements, or 'none'",
  "geometry":    "comma-separated list of geometric shapes, or 'none'",
  "glyphs":      "comma-separated list of symbolic glyphs/motifs, or 'none'",
  "text_found":  "description of text content, or 'none'",
  "letters":     [],
  "words":       [],
  "text_script": "none"
}
"""


def _build_pass2_prompt(p1: dict) -> str:
    """Pass 2 — re-examine image + all Pass 1 findings, output certain / ambiguous."""
    fig  = p1.get("figures",    "none")
    geo  = p1.get("geometry",   "none")
    glyph = p1.get("glyphs",    "none")
    text = p1.get("text_found", "none")
    return f"""You just examined this graffiti mark and found:

  Figures  : {fig}
  Geometry : {geo}
  Glyphs   : {glyph}
  Text     : {text}

Now look at the image again carefully. Based on what you see right now,
decide what is CERTAIN versus AMBIGUOUS across ALL these categories combined.

  CERTAIN  — elements clearly and unambiguously present in the image
  AMBIGUOUS — elements that might be present but are unclear, partial,
               distorted, or could be read multiple ways

Respond with ONLY a JSON object:
{{
  "certain": "comma-separated list of all elements clearly present, e.g. 'circle, drip, letter R, face'",
  "ambiguous": "comma-separated list of uncertain elements, e.g. 'heart, oval, letter K'"
}}

Leave a field as empty string "" if nothing belongs there.
"""


def render_report(stem: str, img_path: Path,
                  p1: dict, certain_text: str, ambiguous_text: str,
                  certain_dets: list[dict], ambiguous_dets: list[dict],
                  letters: list, words: list, text_script: str) -> None:
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
    mask_arr  = load_mask_array(stem, RAS_DIR)

    panel_w = 480
    canvas  = Image.new("RGB", (display_w + panel_w, display_h), (30, 30, 35))
    canvas.paste(mark_img, (0, 0))

    all_dets    = certain_dets + ambiguous_dets
    mark_region = canvas.crop((0, 0, display_w, display_h))
    for i, det in enumerate(all_dets):
        colour = DOT_COLOURS[i % len(DOT_COLOURS)]
        if mask_arr is not None:
            draw_pixel_overlay(mark_region, mask_arr, det["bbox"], colour)
        else:
            d = ImageDraw.Draw(mark_region)
            b = det["bbox"]
            d.rectangle([int(b[0]*display_w), int(b[1]*display_h),
                         int(b[2]*display_w), int(b[3]*display_h)],
                        outline=colour, width=3)
    canvas.paste(mark_region, (0, 0))

    draw = ImageDraw.Draw(canvas)
    for i, det in enumerate(all_dets):
        colour = DOT_COLOURS[i % len(DOT_COLOURS)]
        b  = det["bbox"]
        cx = int(((b[0]+b[2])/2) * display_w)
        cy = int(((b[1]+b[3])/2) * display_h)
        r  = 7
        draw.ellipse([cx-r-1, cy-r-1, cx+r+1, cy+r+1], fill=(0,0,0))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=colour)
        draw.text((cx-4, cy-6), str(i+1), fill=(0,0,0))

    x0 = display_w + 14
    y  = 12

    def txt(text, colour=(220, 220, 220)):
        nonlocal y
        draw.text((x0, y), text, fill=colour)
        y += 18

    def gap(px=6):
        nonlocal y
        y += px

    txt(f"{stem}  [combined]", colour=(255, 255, 100))
    gap()

    # Pass 1 channels
    cat_colours = {
        "figures":  (255, 160, 140),
        "geometry": (140, 200, 255),
        "glyphs":   (200, 255, 140),
    }
    for cat, col in cat_colours.items():
        val = p1.get(cat, "none")
        if val and val != "none":
            txt(f"{cat.upper()}: {val[:50]}{'…' if len(val)>50 else ''}", colour=col)
    txt_val = p1.get("text_found", "none")
    if txt_val and txt_val != "none":
        txt(f"TEXT: {txt_val[:50]}{'…' if len(txt_val)>50 else ''}", colour=(180, 220, 255))
    gap()

    txt("CERTAIN:", colour=(80, 220, 80))
    if certain_dets:
        for i, det in enumerate(certain_dets):
            colour = DOT_COLOURS[i % len(DOT_COLOURS)]
            txt(f"  {i+1}. {SYMBOL_DISPLAY[det['key']]}  {det['confidence']}%", colour=colour)
    elif certain_text:
        for line in word_wrap(certain_text, 50):
            txt(f"  {line}", colour=(160, 220, 160))
    else:
        txt("  (none)", colour=(150, 150, 150))
    gap()

    txt("AMBIGUOUS:", colour=(220, 180, 60))
    offset = len(certain_dets)
    if ambiguous_dets:
        for j, det in enumerate(ambiguous_dets):
            colour = DOT_COLOURS[(offset+j) % len(DOT_COLOURS)]
            txt(f"  {offset+j+1}. {SYMBOL_DISPLAY[det['key']]}  {det['confidence']}%  (?)",
                colour=colour)
    elif ambiguous_text:
        for line in word_wrap(ambiguous_text, 50):
            txt(f"  {line}", colour=(200, 170, 100))
    else:
        txt("  (none)", colour=(150, 150, 150))
    gap()

    txt("TEXT:", colour=(120, 180, 255))
    if words:
        txt(f"  Words:   {' | '.join(words)}", colour=(180, 220, 255))
    if letters:
        txt(f"  Letters: {'  '.join(letters)}", colour=(180, 220, 255))
    if text_script and text_script != "none":
        txt(f"  Script:  {text_script}", colour=(180, 220, 255))
    if not words and not letters:
        txt("  (no text detected)", colour=(150, 150, 150))

    canvas.save(OUT_DIR / f"{stem}_report.png")


def process_mark(stem: str, model: str) -> dict | None:
    img_path = load_mark_image(stem, RAS_DIR, SEG_DIR)
    if img_path is None:
        print(f"  ✗ No image for {stem}")
        return None

    t0 = time.time()
    try:
        img_b64 = image_to_base64(img_path)

        print(f"  {stem}  — pass 1 (combined channels)…")
        raw1    = call_ollama(img_b64, _build_pass1_prompt(), model)
        p1      = parse_json(raw1)
        letters = p1.get("letters", []) or []
        words   = p1.get("words",   []) or []
        script  = p1.get("text_script", "none") or "none"

        print(f"  {stem}  — pass 2 (consolidate)…")
        raw2           = call_ollama(img_b64, _build_pass2_prompt(p1), model)
        p2             = parse_json(raw2)
        certain_text   = p2.get("certain",   "") or ""
        ambiguous_text = p2.get("ambiguous", "") or ""
        if isinstance(certain_text,  list): certain_text  = ", ".join(certain_text)
        if isinstance(ambiguous_text, list): ambiguous_text = ", ".join(ambiguous_text)

        print(f"    → certain:   {certain_text[:80]}{'…' if len(certain_text)>80 else ''}")
        print(f"    → ambiguous: {ambiguous_text[:80]}{'…' if len(ambiguous_text)>80 else ''}")

        certain_dets   = grep_symbols(certain_text)
        ambiguous_dets = grep_symbols(ambiguous_text)
        print(f"    → certain keys:   {[d['key'] for d in certain_dets] or 'none'}")
        print(f"    → ambiguous keys: {[d['key'] for d in ambiguous_dets] or 'none'}")

    except Exception as e:
        print(f"  ✗ {stem}: {e}")
        return None

    elapsed = time.time() - t0
    print(f"  {stem}  ({elapsed:.1f}s)")

    render_report(stem, img_path, p1, certain_text, ambiguous_text,
                  certain_dets, ambiguous_dets, letters, words, script)

    certain_map   = {d["key"]: d["confidence"] for d in certain_dets}
    ambiguous_map = {d["key"]: d["confidence"] for d in ambiguous_dets}
    letters_str   = " ".join(str(l) for l in letters)
    words_str     = " | ".join(str(w) for w in words)

    summary = {
        "stem":           stem,
        "p1_figures":     p1.get("figures",    ""),
        "p1_geometry":    p1.get("geometry",   ""),
        "p1_glyphs":      p1.get("glyphs",     ""),
        "p1_text":        p1.get("text_found", ""),
        "certain_text":   certain_text,
        "ambiguous_text": ambiguous_text,
        "letters":        letters_str,
        "words":          words_str,
        "text_script":    script,
        "n_certain":      len(certain_dets),
        "n_ambiguous":    len(ambiguous_dets),
        "has_text":       int(bool(words or letters)),
    }
    for key in SYMBOL_KEYS:
        summary[f"certain_{key}"]   = certain_map.get(key, 0)
        summary[f"ambiguous_{key}"] = ambiguous_map.get(key, 0)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=f"VLM {VARIANT} channelled detection: certain + ambiguous tiers")
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

    _sum_cols = (["stem", "p1_figures", "p1_geometry", "p1_glyphs", "p1_text",
                  "certain_text", "ambiguous_text", "letters", "words", "text_script",
                  "n_certain", "n_ambiguous", "has_text"]
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
