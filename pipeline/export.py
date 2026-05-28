"""
export.py — Stage public-facing outputs for the website.

Reads from data/ and writes a clean public bundle into outputs/site/:

  outputs/site/
    pca_full.svg           — Full PCA scatter, all tags (website-ready SVG)
    pca_preview.svg        — Landing-page PCA preview with family accents
    dendrogram_full.svg    — Full hierarchical clustering dendrogram (website-ready SVG)
    dendrogram_preview.svg — Simplified 3-rep-per-family dendrogram preview
    metrics.csv            — Clean public metrics table, one row per tag
    vectors.zip            — All public vector traces (SVG, one per tag)
    README.md              — What the downloads are and how to use them
    METHODS.md             — How the analysis was done

Run after cluster.py:
    conda activate othercodes
    python3 pipeline/export.py
"""

import csv
import shutil
import zipfile
from datetime import date
from pathlib import Path

ROOT      = Path(__file__).parent.parent
PLOTS_DIR = ROOT / "data" / "plots"
VEC_DIR   = ROOT / "data" / "vectors"
CSV_IN    = ROOT / "data" / "features.csv"
OUT_DIR   = ROOT / "outputs" / "site"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics included in PCA / clustering (mirrors cluster.py CLUSTER_FEATURES)
CLUSTER_FEATURES = {
    "aspect_ratio", "fill_density", "compactness", "solidity",
    "euler_number", "eccentricity", "n_components", "stroke_width_cv",
    "stroke_width_norm", "skeleton_density", "skeleton_to_area",
    "skeleton_to_perimeter", "branching_density", "n_loops_est",
    "endpoint_branch_ratio", "svg_closed_ratio", "perimeter_norm",
}


# ── 1. SVG plots ───────────────────────────────────────────────────────────────

_PLOT_FILES = (
    "pca_full.svg",
    "pca_preview.svg",
    "dendrogram_full.svg",
    "dendrogram_preview.svg",
)

def export_plots():
    ok = True
    for name in _PLOT_FILES:
        src = PLOTS_DIR / name
        if not src.exists():
            print(f"  ✗ Missing plot: {src}  (run cluster.py first)")
            ok = False
            continue
        shutil.copy2(src, OUT_DIR / name)
        print(f"  ✓ {name}")
    return ok


# ── 2. Clean metrics CSV ───────────────────────────────────────────────────────

def export_metrics():
    if not CSV_IN.exists():
        print(f"  ✗ Missing features CSV: {CSV_IN}  (run analyze.py first)")
        return False

    with open(CSV_IN, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("  ✗ features.csv is empty")
        return False

    # Build public column order:
    # tag_id | source_filename | <all metrics> | used_in_clustering
    internal_cols = set(rows[0].keys())
    metric_cols   = [c for c in rows[0].keys() if c != "stem"]
    out_cols      = ["tag_id", "source_filename"] + metric_cols + ["used_in_clustering"]

    out_path = OUT_DIR / "metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for row in rows:
            stem = row["stem"]
            out_row = {
                "tag_id":            stem,
                "source_filename":   stem + ".jpg",
                "used_in_clustering": "",   # filled per column below
            }
            for col in metric_cols:
                out_row[col] = row.get(col, "")
            # Mark clustering flag per row (same value for all rows — put in header note instead)
            out_row["used_in_clustering"] = "|".join(
                c for c in metric_cols if c in CLUSTER_FEATURES
            ) if stem == rows[0]["stem"] else ""
            writer.writerow(out_row)

    # Simpler approach: add a boolean column per metric row
    # Rewrite with a cleaner structure
    with open(out_path, "w", newline="") as f:
        fieldnames = ["tag_id", "source_filename"] + metric_cols + ["used_in_clustering"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stem = row["stem"]
            out_row = {"tag_id": stem, "source_filename": stem + ".jpg"}
            for col in metric_cols:
                out_row[col] = row.get(col, "")
            out_row["used_in_clustering"] = "see METHODS.md"
            writer.writerow(out_row)

    print(f"  ✓ metrics.csv  ({len(rows)} tags, {len(metric_cols)} metrics)")
    return True


# ── 3. Vectors ZIP ─────────────────────────────────────────────────────────────

def export_vectors_zip():
    # Include only *_web.svg files — clean public-facing traces, no background
    web_svgs = sorted(VEC_DIR.glob("*_web.svg"))
    if not web_svgs:
        print(f"  ✗ No web SVGs found in {VEC_DIR}  (run analyze.py first)")
        return False

    zip_path = OUT_DIR / "vectors.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for svg in web_svgs:
            # Strip _web suffix for the public filename: IMG_4491_web.svg → IMG_4491.svg
            public_name = svg.stem.replace("_web", "") + ".svg"
            zf.write(svg, arcname=f"vectors/{public_name}")

    total_kb = zip_path.stat().st_size // 1024
    print(f"  ✓ vectors.zip  ({len(web_svgs)} files, {total_kb} KB)")
    return True


# ── 4. README.md ───────────────────────────────────────────────────────────────

README = """\
# other.codes — Public Dataset

This folder contains downloadable outputs from the other.codes graffiti tag
analysis project. Files are generated automatically from the analysis pipeline
and updated as new tags are collected and processed.

## Visualisations

### pca_vector.svg
A PCA (Principal Component Analysis) scatter plot placing each tag in a
two-dimensional style space derived from morphological measurements of its
shape, stroke character, and topology. Tags that appear close together are
more similar in measured style. Each data point is rendered as the actual
vector trace of that tag rather than an abstract marker.

### dendrogram.svg
A hierarchical clustering dendrogram showing the similarity structure of the
full tag collection. Tags joined by shorter branches are more similar to each
other. The tree is computed using Ward linkage (Euclidean distance) on the
same 17 standardised features as the PCA. Coloured strips below the glyph
row indicate the three broadest style families identified by the analysis.

## Downloads

### vectors.zip
A ZIP archive containing one SVG vector trace per tag. Each SVG is a clean
black-stroke vector generated by tracing the binary segmentation mask of the
original photograph using vtracer (spline mode). The traces are scale-
normalised — they represent letterform shape, not physical size.

### metrics.csv
A table of morphological measurements, one row per tag. Covers shape (area,
aspect ratio, compactness, solidity), stroke width (mean, variation), skeleton
topology (branching, loops, endpoints), and vector complexity (path count,
closed-path ratio). See METHODS.md for a full description of every column.

### METHODS.md
Full description of the analysis pipeline: how tags are segmented, vectorised,
measured, and compared. Includes all metric definitions, preprocessing steps,
PCA settings, and clustering parameters.

### pca_interpretation.md
A plain-language interpretation of the two principal components shown in
pca_vector.svg — which morphological features drive each axis and what they
reveal about style. Generated automatically from the loadings of the current
dataset; updates whenever new tags are added.

## Dataset notes

This is an early public dataset from a prototype collection run. The sample
is small and geographically limited. Results should be treated as exploratory
rather than definitive. Analysis code is available at
[github.com/untoldlabs/otherCodes](https://github.com/untoldlabs/otherCodes).

## Citation

> other.codes graffiti tag dataset, Untold Labs (https://other.codes), {year}.

[other.codes](https://other.codes) — built by Untold Labs.
""".format(year=date.today().year)


# ── 5. METHODS.md ──────────────────────────────────────────────────────────────

METHODS = """\
# other.codes — Analysis Methods

This document describes how the analysis pipeline converts raw field photographs
into the published metrics, vector traces, and visualisations.

## Pipeline overview

1. **Photograph** — Tags are photographed in the field using an iPhone.
   Images are captured as HEIC and converted to JPEG for processing.

2. **Segmentation** — Each photograph is opened in a browser-based annotation
   tool (Flask, port 5050). The tag is isolated from the background using either:
   - SAM2 (Segment Anything Model 2, Hiera Large) with point or box prompts, or
   - a Random Forest pixel classifier trained on brush-annotated foreground /
     background strokes (multi-scale features, ilastik-style).
   The output is a binary mask (white = tag, black = background) and an RGBA
   isolated crop, saved as `{stem}_mask.png` and `{stem}_isolated.png`.

3. **Rasterisation** — The mask is cropped to its tight bounding box and saved
   to `data/rasters/` as a clean binary PNG. This removes irrelevant background
   pixels and ensures all shape metrics are computed on the tag alone.

4. **Vectorisation** — The cropped mask is inverted (black tag on white ground)
   and passed to vtracer (spline mode, binary colourmode, filter\_speckle=4)
   via its Python API. The result is an SVG path trace of the tag outline,
   saved to `data/vectors/`. These are the files distributed in vectors.zip.

5. **Feature extraction** — Around 30 morphological metrics are computed from
   the binary mask for each tag. See the Metrics section below for definitions.

6. **PCA and clustering** — The 17 scale-independent metrics listed below are
   standardised and passed to PCA (for the scatter plot) and Ward hierarchical
   clustering (for the dendrogram). See the Analysis settings section.

## Metrics

All metrics are computed from the binary mask unless otherwise noted.
Normalisation uses √area as the scale factor so that results are independent
of image resolution and physical tag size.

### Shape

| Metric | Description |
|---|---|
| area\_px | Total foreground pixel count |
| bbox\_w / bbox\_h | Bounding box width and height (pixels) |
| aspect\_ratio | bbox\_w / bbox\_h |
| fill\_density | area\_px / (bbox\_w × bbox\_h) |
| compactness | 4π × area / perimeter² (1.0 = perfect circle) |
| solidity | area / convex hull area (largest connected component) |
| euler\_number | Topological proxy for enclosed holes (e.g. O, A, 4 have holes) |
| eccentricity | Eccentricity of best-fit ellipse (largest connected component) |
| n\_components | Number of disconnected foreground regions |
| perimeter | Total contour length (pixels) |
| perimeter\_norm | perimeter / √area |

### Stroke width

Stroke width is estimated per skeleton pixel as twice the Euclidean distance
transform value at that point — i.e. the diameter of the largest circle that
fits inside the stroke at that location.

| Metric | Description |
|---|---|
| mean\_stroke\_width | Mean stroke diameter across all skeleton pixels |
| stroke\_width\_std | Standard deviation of stroke diameter |
| stroke\_width\_cv | Coefficient of variation: std / mean |
| stroke\_width\_norm | mean\_stroke\_width / √area |
| max\_stroke\_width | Maximum stroke diameter |

### Skeleton topology

The skeleton (medial axis) is computed with `skimage.morphology.skeletonize`.
Branch points and endpoints are identified by counting 8-connected neighbours
of each skeleton pixel.

| Metric | Description |
|---|---|
| skeleton\_length | Total skeleton pixel count |
| skeleton\_density | skeleton\_length / √area |
| skeleton\_to\_area | skeleton\_length / area\_px |
| skeleton\_to\_perimeter | skeleton\_length / perimeter |
| n\_branch\_points | Skeleton pixels with ≥ 3 neighbours |
| n\_endpoints | Skeleton pixels with exactly 1 neighbour |
| branching\_density | n\_branch\_points / skeleton\_length |
| n\_loops\_est | max(0, branch\_pts − endpoints + 1) — Euler-characteristic loop estimate |
| endpoint\_branch\_ratio | n\_endpoints / n\_branch\_points |

### Vector complexity

Measured from the vtracer SVG output, not from the raster mask.

| Metric | Description |
|---|---|
| n\_svg\_paths | Number of SVG `<path>` elements |
| svg\_path\_complexity | Total path command count across all paths |
| svg\_closed\_paths | Number of paths ending with a Z (close) command |
| svg\_closed\_ratio | svg\_closed\_paths / n\_svg\_paths |

## Metrics used for clustering and PCA

The following 17 metrics are passed to PCA and to the hierarchical clustering
that produces the dendrogram. They are selected for being scale-independent
and not directly redundant with each other. All remaining metrics are retained
in the public metrics.csv for reference but do not influence the analysis.

```
aspect_ratio         fill_density         compactness
solidity             euler_number         eccentricity
n_components         stroke_width_cv      stroke_width_norm
skeleton_density     skeleton_to_area     skeleton_to_perimeter
branching_density    n_loops_est          endpoint_branch_ratio
svg_closed_ratio     perimeter_norm
```

## Analysis settings

### Preprocessing

All 17 clustering metrics are standardised to zero mean and unit variance using
`sklearn.preprocessing.StandardScaler` before any analysis. This ensures that
features measured in different units (pixels, ratios, counts) contribute equally.

### PCA

| Setting | Value |
|---|---|
| Library | `sklearn.decomposition.PCA` |
| n\_components | min(n\_tags, n\_features, 10) |
| Input | StandardScaler-normalised feature matrix |
| Plot | PC1 (horizontal axis) × PC2 (vertical axis) |

Variance explained by each PC is shown on the axis labels of pca\_vector.svg
and in pca\_interpretation.md.

### Hierarchical clustering (dendrogram)

| Setting | Value |
|---|---|
| Library | `scipy.cluster.hierarchy.linkage` + `dendrogram` |
| Linkage method | Ward |
| Distance metric | Euclidean (on standardised features) |
| Family grouping | `scipy.cluster.hierarchy.fcluster`, 3 groups, maxclust criterion |

The three family groups shown as coloured strips in dendrogram.svg are
determined by cutting the Ward tree into exactly 3 clusters using `fcluster`.

## Software

| Package | Role |
|---|---|
| Python 3.11 | Runtime |
| numpy | Numerical arrays |
| scipy | Distance transform, clustering, dendrogram |
| scikit-image | Skeletonize, regionprops, contour tracing |
| scikit-learn | StandardScaler, PCA |
| Pillow | Image I/O |
| vtracer 0.6.x | Raster → SVG vectorisation |
| Flask | Browser-based annotation tool |
| SAM2 (Meta) | Segment Anything Model 2 — point/box-prompted segmentation |
| matplotlib | Reference PCA scatter (raster thumbnails, internal use) |

## Limitations

- **Small dataset** — the current collection contains approximately 12 tags from
  a single session. Results are exploratory and should not be generalised.
- **Single collector** — all photographs were taken by one person in one city.
  Geographic and stylistic coverage is narrow at this stage.
- **Manual segmentation** — masks are hand-annotated and may contain errors,
  particularly where backgrounds are complex or marks are faint.
- **Scale invariance** — all metrics are normalised to be independent of image
  resolution and tag size. Absolute scale information (physical size of the tag
  in the real world) is not captured.
- **No temporal or geographic metadata** — the public dataset does not include
  location, date, or photographer information.
"""


def export_docs():
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")
    (OUT_DIR / "METHODS.md").write_text(METHODS, encoding="utf-8")
    print("  ✓ README.md")
    print("  ✓ METHODS.md")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\nStaging public outputs → {OUT_DIR}\n")
    ok = True
    ok &= export_plots()
    ok &= export_metrics()
    ok &= export_vectors_zip()
    export_docs()

    if ok:
        print(f"\n✓ outputs/site/ ready — run ./scripts/export_for_site.sh to deploy\n")
    else:
        print(f"\n⚠ Some outputs missing — check warnings above\n")


if __name__ == "__main__":
    main()
