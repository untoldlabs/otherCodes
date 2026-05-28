"""
analyze.py — Feature extraction for segmented symbolic mark tags.

For each tag in <project>/data/segmented/ this script:
  1. Loads the binary mask
  2. Computes raster shape features (area, solidity, compactness, etc.)
  3. Skeletonises the mask and extracts stroke/topology features
  4. Runs vtracer to vectorise → <project>/data/vectors/{stem}.svg
  5. Parses the SVG for vector complexity features
  6. Writes everything to <project>/data/features.csv (one row per tag)

Usage:
    conda activate othercodes
    python3 pipeline/analyze.py --project /path/to/project
    python3 pipeline/analyze.py          # uses repo data/ as project root
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import vtracer

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import convolve
from skimage import measure
from skimage.morphology import skeletonize


# ── SVG helpers ────────────────────────────────────────────────────────────────

def _strip_white_fills(svg_text: str) -> str:
    """Remove SVG elements with white/light fills — vtracer background artefacts."""
    svg_text = re.sub(
        r'<(?:rect|path|polygon)\b[^>]*\bfill="#(?:fff|ffffff|FFF|FFFFFF)"[^>]*/?>',
        '', svg_text, flags=re.IGNORECASE
    )
    svg_text = re.sub(
        r'<(?:rect|path|polygon)\b[^>]*\bfill="white"[^>]*/?>',
        '', svg_text, flags=re.IGNORECASE
    )
    return svg_text


# ── Paths (set in main() from --project arg) ───────────────────────────────────
SEG_DIR = VEC_DIR = RAS_DIR = CSV_OUT = None

def _init_paths(project_root: Path):
    global SEG_DIR, VEC_DIR, RAS_DIR, CSV_OUT
    SEG_DIR = project_root / "data" / "segmented"
    VEC_DIR = project_root / "data" / "vectors"
    RAS_DIR = project_root / "data" / "rasters"
    CSV_OUT = project_root / "data" / "features.csv"
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    RAS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature extraction ─────────────────────────────────────────────────────────

def analyze_tag(stem: str) -> dict | None:
    mask_path     = SEG_DIR / f"{stem}_mask.png"
    isolated_path = SEG_DIR / f"{stem}_isolated.png"

    if not mask_path.exists():
        print(f"  ✗ No mask for {stem}")
        return None

    # Load binary mask
    mask = np.array(Image.open(mask_path).convert("L")) > 128
    if mask.sum() == 0:
        print(f"  ✗ Empty mask for {stem}")
        return None

    feat = {"stem": stem}

    # ── 1. Crop to bounding box ────────────────────────────────────────────────
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    m = mask[rmin:rmax+1, cmin:cmax+1]  # cropped mask

    h, w  = m.shape
    area  = int(m.sum())
    scale = float(np.sqrt(area)) + 1e-6   # for normalisation

    # ── Save cropped raster outputs ────────────────────────────────────────────
    # Binary mask crop (white tag on black)
    ras_path = RAS_DIR / f"{stem}_mask_crop.png"
    Image.fromarray((m * 255).astype(np.uint8), mode="L").save(ras_path)

    # Isolated crop (RGBA tag on transparent background)
    if isolated_path.exists():
        iso_crop_path = RAS_DIR / f"{stem}_isolated_crop.png"
        iso_full = np.array(Image.open(isolated_path).convert("RGBA"))
        iso_crop = iso_full[rmin:rmax+1, cmin:cmax+1]
        Image.fromarray(iso_crop, mode="RGBA").save(iso_crop_path)

    # Skeleton raster — saved after skeletonize() below, see step 5

    # ── 2. Basic shape ─────────────────────────────────────────────────────────
    feat["area_px"]      = area
    feat["bbox_w"]       = w
    feat["bbox_h"]       = h
    feat["aspect_ratio"] = round(w / h, 4)
    feat["fill_density"] = round(area / (w * h), 4)

    # ── 3. Contour / perimeter / compactness ───────────────────────────────────
    contours  = measure.find_contours(m.astype(float), 0.5)
    perimeter = int(sum(len(c) for c in contours)) if contours else 0
    feat["perimeter"]    = perimeter
    feat["compactness"]  = round(4 * np.pi * area / (perimeter**2), 4) \
                           if perimeter > 0 else 0
    feat["perimeter_norm"] = round(perimeter / scale, 4)

    # ── 4. Region properties ───────────────────────────────────────────────────
    labeled = measure.label(m)
    props   = measure.regionprops(labeled)
    feat["n_components"] = len(props)

    if props:
        largest = max(props, key=lambda p: p.area)
        feat["solidity"]      = round(largest.solidity, 4)
        feat["euler_number"]  = int(largest.euler_number)  # holes/loops topology
        feat["eccentricity"]  = round(largest.eccentricity, 4)
    else:
        feat["solidity"]     = 0
        feat["euler_number"] = 0
        feat["eccentricity"] = 0

    # ── 5. Distance transform → stroke width ──────────────────────────────────
    dist = ndimage.distance_transform_edt(m)
    skel = skeletonize(m)

    # Save skeleton raster (1-px centrelines, white on black, cropped bbox)
    skel_ras_path = RAS_DIR / f"{stem}_skeleton.png"
    Image.fromarray((skel.astype(np.uint8) * 255), mode="L").save(skel_ras_path)

    skel_vals = dist[skel]   # distance values at skeleton pixels = stroke radius
    if len(skel_vals) > 0:
        feat["mean_stroke_width"] = round(float(np.mean(skel_vals)) * 2, 2)   # radius→diameter
        feat["stroke_width_std"]  = round(float(np.std(skel_vals))  * 2, 2)
        feat["stroke_width_cv"]   = round(
            float(np.std(skel_vals)) / (float(np.mean(skel_vals)) + 1e-6), 4)
        feat["stroke_width_norm"] = round(feat["mean_stroke_width"] / scale, 4)
        feat["max_stroke_width"]  = round(float(np.max(skel_vals)) * 2, 2)
    else:
        feat["mean_stroke_width"] = 0
        feat["stroke_width_std"]  = 0
        feat["stroke_width_cv"]   = 0
        feat["stroke_width_norm"] = 0
        feat["max_stroke_width"]  = 0

    # ── 6. Skeleton metrics ────────────────────────────────────────────────────
    skel_length = int(skel.sum())
    feat["skeleton_length"]       = skel_length
    feat["skeleton_density"]      = round(skel_length / scale, 4)
    feat["skeleton_to_area"]      = round(skel_length / area, 4)
    feat["skeleton_to_perimeter"] = round(skel_length / (perimeter + 1e-6), 4)

    # ── 7. Skeleton topology: branch points, endpoints, loops ─────────────────
    # Count 8-connected neighbours of each skeleton pixel
    kernel        = np.ones((3, 3), dtype=int)
    kernel[1, 1]  = 0
    neighbour_ct  = convolve(skel.astype(int), kernel, mode='constant', cval=0)

    branch_pts    = int((skel & (neighbour_ct >= 3)).sum())
    endpoints     = int((skel & (neighbour_ct == 1)).sum())

    feat["n_branch_points"]  = branch_pts
    feat["n_endpoints"]      = endpoints
    feat["branching_density"]= round(branch_pts / (skel_length + 1e-6), 4)
    # Rough loop estimate from Euler characteristic of skeleton graph
    feat["n_loops_est"]      = max(0, branch_pts - endpoints + 1)
    # Endpoint:branch ratio — high = flowing/cursive, low = intersecting/complex
    feat["endpoint_branch_ratio"] = round(
        endpoints / (branch_pts + 1e-6), 4)

    # ── 8. Vectorise with vtracer (Python API — no CLI needed) ────────────────
    # Use the mask_crop PNG (clean binary grayscale) — RGBA isolated PNGs cause
    # vtracer to fail silently in binary mode due to the alpha channel.
    svg_path = VEC_DIR / f"{stem}.svg"
    if not svg_path.exists():
        try:
            vtracer.convert_image_to_svg_py(
                str(ras_path),      # mask_crop.png — white tag on black, no alpha
                str(svg_path),      # output path (writes directly to file)
                colormode="binary",
                filter_speckle=4,
                mode="spline",
            )
            if svg_path.exists() and svg_path.stat().st_size > 50:
                print(f"  ✓ Vectorised {stem}  ({svg_path.stat().st_size} bytes)")
            else:
                print(f"  ✗ vtracer wrote empty file for {stem}")
        except Exception as e:
            print(f"  ✗ vtracer failed for {stem}: {e}")

    # ── 8b. Web SVG: black tag on transparent background ─────────────────────
    # Inverted mask (black tag on white) → vtracer traces the tag, not the background.
    # Strip any residual white fills → result is black paths on transparent SVG canvas.
    web_svg_path = VEC_DIR / f"{stem}_web.svg"
    if not web_svg_path.exists():
        try:
            m_inv = ((1 - m.astype(np.uint8)) * 255).astype(np.uint8)
            tmp_web = RAS_DIR / f"_tmp_{stem}_web.png"
            Image.fromarray(m_inv, mode="L").save(tmp_web)
            vtracer.convert_image_to_svg_py(
                str(tmp_web),
                str(web_svg_path),
                colormode="binary",
                filter_speckle=4,
                mode="spline",
            )
            tmp_web.unlink(missing_ok=True)
            # Strip white background fills vtracer may have added
            if web_svg_path.exists():
                cleaned = _strip_white_fills(web_svg_path.read_text(encoding="utf-8"))
                web_svg_path.write_text(cleaned, encoding="utf-8")
                print(f"  ✓ Web SVG generated {stem}  ({web_svg_path.stat().st_size} bytes)")
        except Exception as e:
            print(f"  ✗ Web SVG failed for {stem}: {e}")

    # ── 8c. Vectorise skeleton ────────────────────────────────────────────────
    # Dilate the 1-px skeleton to give vtracer enough width to trace cleanly.
    skel_svg_path = VEC_DIR / f"{stem}_skeleton.svg"
    if not skel_svg_path.exists():
        try:
            skel_dilated = ndimage.binary_dilation(skel, iterations=3)
            tmp_path = RAS_DIR / f"_tmp_{stem}_skel.png"
            Image.fromarray((skel_dilated.astype(np.uint8) * 255), mode="L").save(tmp_path)
            vtracer.convert_image_to_svg_py(
                str(tmp_path),
                str(skel_svg_path),     # output path (writes directly to file)
                colormode="binary",
                filter_speckle=2,
                mode="spline",
            )
            tmp_path.unlink(missing_ok=True)
            print(f"  ✓ Skeleton vectorised {stem}")
        except Exception as e:
            print(f"  ✗ Skeleton vtracer failed for {stem}: {e}")

    # ── 9. Parse SVG for vector features ──────────────────────────────────────
    if svg_path.exists():
        try:
            tree  = ET.parse(svg_path)
            root  = tree.getroot()
            # Handle SVG namespace
            ns    = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
            tag_p = f"{{{ns}}}path" if ns else "path"
            paths = root.findall(f".//{tag_p}")

            n_paths   = len(paths)
            n_nodes   = sum(
                len([t for t in p.get('d', '').split()
                     if t and t[0].isalpha()])
                for p in paths)
            n_closed  = sum(
                1 for p in paths
                if p.get('d', '').strip().upper().endswith('Z'))

            feat["n_svg_paths"]        = n_paths
            feat["svg_path_complexity"]= n_nodes
            feat["svg_closed_paths"]   = n_closed
            feat["svg_closed_ratio"]   = round(n_closed / (n_paths + 1e-6), 4)
        except Exception as e:
            print(f"  ✗ SVG parse failed for {stem}: {e}")
            feat["n_svg_paths"] = feat["svg_path_complexity"] = \
            feat["svg_closed_paths"] = feat["svg_closed_ratio"] = 0
    else:
        feat["n_svg_paths"] = feat["svg_path_complexity"] = \
        feat["svg_closed_paths"] = feat["svg_closed_ratio"] = 0

    return feat


# ── Run all ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract features from segmented tags")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project root directory (contains data/segmented/). "
                             "Defaults to the repo data/ directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    project_root = args.project.expanduser().resolve() if args.project else repo_root
    _init_paths(project_root)
    print(f"\nProject: {project_root}")

    stems = sorted({p.stem.replace("_mask", "").replace("_isolated", "")
                    for p in SEG_DIR.glob("*.png")
                    if "_mask" in p.name})

    if not stems:
        print("No segmented masks found in data/segmented/")
        return

    print(f"\nAnalysing {len(stems)} tags...\n")
    rows = []
    for stem in stems:
        print(f"  {stem}")
        feat = analyze_tag(stem)
        if feat:
            rows.append(feat)
            print(f"    area={feat['area_px']}  solidity={feat['solidity']}  "
                  f"stroke_w={feat['mean_stroke_width']}  "
                  f"skel_density={feat['skeleton_density']}  "
                  f"euler={feat['euler_number']}")

    if not rows:
        print("No features extracted.")
        return

    # Write CSV
    cols = list(rows[0].keys())
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n✓ Features saved to {CSV_OUT}")
    print(f"  {len(rows)} tags × {len(cols)-1} features\n")


if __name__ == "__main__":
    main()
