"""
segment.py — Isolate graffiti tags from background using OpenCV.

Strategy:
  1. Convert to LAB colour space (separates lightness from colour)
  2. Use K-means to cluster pixels into foreground/background
  3. Keep the most saturated / non-wall cluster (the tag)
  4. Clean up with morphological operations
  5. Output: transparent PNG + binary mask

Usage:
    python pipeline/segment.py
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SEG_DIR = Path(__file__).parent.parent / "data" / "segmented"
SEG_DIR.mkdir(parents=True, exist_ok=True)


def segment_tag(jpg_path: Path) -> None:
    stem = jpg_path.stem
    out_rgba = SEG_DIR / f"{stem}_isolated.png"
    out_mask = SEG_DIR / f"{stem}_mask.png"

    if out_rgba.exists() and out_mask.exists():
        print(f"  skip (already done): {stem}")
        return

    print(f"  segmenting: {jpg_path.name} ...", end=" ", flush=True)

    img_bgr = cv2.imread(str(jpg_path))
    if img_bgr is None:
        print(f"  ERROR: could not read {jpg_path}")
        return

    h, w = img_bgr.shape[:2]

    # --- Step 1: Resize for processing speed, keep aspect ratio ---
    max_dim = 1024
    scale = min(max_dim / w, max_dim / h, 1.0)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # --- Step 2: Convert to LAB and run K-means ---
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])

    # --- Step 3: Score each cluster by colour saturation (A and B channels in LAB) ---
    # Wall/background tends to be low saturation (near-grey), tags are more saturated
    scores = []
    for i in range(k):
        a_dev = abs(float(centers[i][1]) - 128)  # A channel deviation from neutral
        b_dev = abs(float(centers[i][2]) - 128)  # B channel deviation from neutral
        scores.append(a_dev + b_dev)

    # Also penalise clusters that are very light (the wall)
    for i in range(k):
        lightness = float(centers[i][0])
        if lightness > 200:  # very bright = probably wall
            scores[i] *= 0.3

    # Pick the top 1-2 clusters as foreground
    sorted_clusters = np.argsort(scores)[::-1]
    fg_clusters = sorted_clusters[:2]  # top 2 most saturated clusters

    # Build mask from selected clusters
    mask_small = np.zeros(small.shape[:2], dtype=np.uint8)
    for c in fg_clusters:
        mask_small[labels == c] = 255

    # --- Step 4: Morphological clean-up ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small blobs — keep only the largest connected component(s)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask_small)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        threshold = max(areas) * 0.05  # keep components > 5% of largest
        clean_mask = np.zeros_like(mask_small)
        for i, area in enumerate(areas, start=1):
            if area >= threshold:
                clean_mask[cc_labels == i] = 255
        mask_small = clean_mask

    # --- Step 5: Scale mask back to original image size ---
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- Step 6: Apply Gaussian blur to soften edges ---
    mask_full = cv2.GaussianBlur(mask_full, (5, 5), 0)
    _, mask_full = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)

    # --- Step 7: Save outputs ---
    # Binary mask
    Image.fromarray(mask_full, mode="L").save(out_mask)

    # Transparent RGBA
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([img_rgb, mask_full])
    Image.fromarray(rgba, mode="RGBA").save(out_rgba)

    print("✓")


def main():
    jpg_files = sorted(RAW_DIR.glob("*.jpg"))
    if not jpg_files:
        print("No JPG files found in", RAW_DIR)
        print("Run pipeline/convert.py first.")
        return

    print(f"Segmenting {len(jpg_files)} images...\n")
    for jpg in jpg_files:
        segment_tag(jpg)

    print(f"\nDone. Results saved to {SEG_DIR}")
    print(f"  *_isolated.png = tag on transparent background")
    print(f"  *_mask.png     = binary mask (white = tag)")


if __name__ == "__main__":
    main()
