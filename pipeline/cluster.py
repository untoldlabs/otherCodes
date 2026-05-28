"""
cluster.py — PCA and hierarchical clustering of tag features.

Reads <project>/data/features.csv and produces SVGs in <project>/data/plots/:

  pca_full.svg            — PCA scatter, all tags, vector traces, prominent axes
  pca_preview.svg         — landing-page PCA: larger glyphs, family halos, interpretation
  dendrogram_full.svg     — full hierarchical clustering dendrogram, portrait
  dendrogram_preview.svg  — 3-representative-per-family preview, grouped columns
  pca.svg                 — reference PCA with raster thumbnails (not for site)

Usage:
    conda activate othercodes
    python3 pipeline/cluster.py --project /path/to/project
    python3 pipeline/cluster.py          # uses repo data/ as project root
"""

import argparse
import base64
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths (set in main() from --project arg) ───────────────────────────────────
CSV_IN = RAS_DIR = VEC_DIR = PLOT_DIR = None

def _init_paths(project_root: Path):
    global CSV_IN, RAS_DIR, VEC_DIR, PLOT_DIR
    CSV_IN   = project_root / "data" / "features.csv"
    RAS_DIR  = project_root / "data" / "rasters"
    VEC_DIR  = project_root / "data" / "vectors"
    PLOT_DIR = project_root / "data" / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Features used for clustering — normalised, scale-independent
CLUSTER_FEATURES = [
    "aspect_ratio",
    "fill_density",
    "compactness",
    "solidity",
    "euler_number",
    "eccentricity",
    "n_components",
    "stroke_width_cv",
    "stroke_width_norm",
    "skeleton_density",
    "skeleton_to_area",
    "skeleton_to_perimeter",
    "branching_density",
    "n_loops_est",
    "endpoint_branch_ratio",
    "svg_closed_ratio",
    "perimeter_norm",
]

# ── Default colour palette (used when --colors not specified) ──────────────────
# Enough for up to 10 families; first 3 keep the original brand colours.
DEFAULT_PALETTE = [
    "#FF3EA5",  # 1 hot pink
    "#00B8D9",  # 2 cyan
    "#E6B800",  # 3 acid gold
    "#7B61FF",  # 4 violet
    "#00C896",  # 5 teal
    "#FF6B35",  # 6 orange
    "#A259FF",  # 7 purple
    "#06D6A0",  # 8 mint
    "#EF476F",  # 9 rose
    "#FFD166",  # 10 amber
]

def make_palette(k: int, color_list: list[str] | None = None) -> dict[int, str]:
    """Return {1: color, 2: color, …} for k families."""
    base = color_list if color_list else DEFAULT_PALETTE
    # Cycle if the user supplied fewer colours than k
    return {i + 1: base[i % len(base)] for i in range(k)}

def make_names(k: int) -> dict[int, str]:
    """Return {1: 'Family A', 2: 'Family B', …} up to k."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {i + 1: f"Family {labels[i]}" for i in range(k)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _svg_natural_dims(path: Path) -> tuple[float, float] | None:
    """Return (width, height) from an SVG file's root element, or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        w = re.search(r'<svg[^>]+\bwidth="([0-9.]+)"', text)
        h = re.search(r'<svg[^>]+\bheight="([0-9.]+)"', text)
        if w and h:
            return float(w.group(1)), float(h.group(1))
    except Exception:
        pass
    return None


def _scaled_image_tag(href: str, cx: float, cy: float,
                      max_w: float, max_h: float,
                      nat_dims: tuple[float, float] | None) -> str:
    """Return an SVG <image> element scaled to fit max_w×max_h, correct aspect ratio."""
    if nat_dims:
        nw, nh = nat_dims
        scale  = min(max_w / nw, max_h / nh)
        dw, dh = nw * scale, nh * scale
    else:
        dw, dh = max_w, max_h
    x = cx - dw / 2
    y = cy - dh / 2
    return (f'<image href="{href}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{dw:.1f}" height="{dh:.1f}"/>')


def _glyph_svg(stem: str) -> tuple[str, tuple[float, float] | None] | None:
    """Return (base64_href, nat_dims) for a tag's web SVG, or None if missing."""
    for candidate in [VEC_DIR / f"{stem}_web.svg", VEC_DIR / f"{stem}.svg"]:
        if candidate.exists():
            try:
                nat  = _svg_natural_dims(candidate)
                b64  = base64.b64encode(candidate.read_bytes()).decode("ascii")
                return f"data:image/svg+xml;base64,{b64}", nat
            except Exception:
                pass
    return None


def load_thumbnail(stem: str, size: int = 80) -> np.ndarray | None:
    """Load the cropped isolated PNG and return a small RGBA thumbnail."""
    for candidate in [
        RAS_DIR / f"{stem}_isolated_crop.png",
        RAS_DIR / f"{stem}_mask_crop.png",
    ]:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGBA")
                img.thumbnail((size, size), Image.LANCZOS)
                bg = Image.new("RGBA", img.size, (15, 15, 15, 255))
                bg.paste(img, mask=img.split()[3])
                return np.array(bg.convert("RGB"))
            except Exception:
                continue
    return None


def tag_image_box(stem: str, zoom: float = 0.4) -> AnnotationBbox | None:
    """Return a matplotlib AnnotationBbox for a tag thumbnail, or None."""
    arr = load_thumbnail(stem)
    if arr is None:
        return None
    oi = OffsetImage(arr, zoom=zoom)
    oi.image.axes = None
    ab = AnnotationBbox(oi, (0, 0),
                        frameon=True,
                        bboxprops=dict(edgecolor="#333", linewidth=0.5,
                                       facecolor="white", alpha=0.85))
    return ab


def _pc_brief_labels(loadings: "pd.DataFrame", pc_name: str, n: int = 2
                     ) -> tuple[list[str], list[str]]:
    """Return (pos_names, neg_names) — top-n features driving a PC in each direction."""
    col  = loadings[pc_name].sort_values(ascending=False)
    pos  = [f.replace("_", " ") for f in col.head(n).index]
    neg  = [f.replace("_", " ") for f in col.tail(n).iloc[::-1].index]
    return pos, neg


# ── Reference PCA (matplotlib, raster thumbnails) ─────────────────────────────

def plot_pca(df: pd.DataFrame, X_pca: np.ndarray, explained: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#111")
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=0, alpha=0)
    for i, stem in enumerate(df["stem"]):
        x, y = X_pca[i, 0], X_pca[i, 1]
        ab = tag_image_box(stem, zoom=0.35)
        if ab is not None:
            ab.xy = (x, y)
            ab.xybox = (x, y)
            ab.xycoords = "data"
            ab.boxcoords = "data"
            ax.add_artist(ab)
        else:
            ax.text(x, y, stem, fontsize=7, color="#aaa", ha='center', va='center')
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", color="#aaa", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", color="#aaa", fontsize=11)
    ax.set_title("Tag style space — PCA", color="#fff", fontsize=14, pad=12)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    plt.tight_layout()
    out = PLOT_DIR / "pca.svg"
    plt.savefig(out, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Reference PCA saved to {out}")


# ── Full PCA scatter — all tags, pure SVG ─────────────────────────────────────

def plot_pca_full(df: pd.DataFrame, X_pca: np.ndarray, explained: np.ndarray):
    """Full PCA scatter: all tags as SVG traces, prominent axes, tick scores.

    Suitable for pan/zoom in JavaScript. Output: data/plots/pca_full.svg
    """
    W, H   = 2400, 1800
    MARGIN = 200
    GLYPH  = 160

    x_vals = X_pca[:, 0]
    y_vals = X_pca[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    def to_px(xi, yi):
        px = MARGIN + (xi - x_min) / x_range * (W - 2 * MARGIN)
        py = H - MARGIN - (yi - y_min) / y_range * (H - 2 * MARGIN)
        return px, py

    LINE_COLOR = "#000000"
    AXIS_OP    = 0.70
    AXIS_W     = 2.5
    TICK_LEN   = 14
    ARROW      = 14
    LABEL_FS   = 34
    N_TICKS    = 5
    TICK_FS    = 22

    pc1_label = f"PC1 ({explained[0]:.1f}%)"
    pc2_label = f"PC2 ({explained[1]:.1f}%)"

    ax_x0, ax_x1 = MARGIN, W - MARGIN
    ax_y0, ax_y1 = H - MARGIN, MARGIN

    arrow_x = (f"{ax_x1},{ax_y0} "
               f"{ax_x1-ARROW},{ax_y0-ARROW//2} "
               f"{ax_x1-ARROW},{ax_y0+ARROW//2}")
    arrow_y = (f"{ax_x0},{ax_y1} "
               f"{ax_x0-ARROW//2},{ax_y1+ARROW} "
               f"{ax_x0+ARROW//2},{ax_y1+ARROW}")

    cx_mid = (ax_x0 + ax_x1) // 2
    cy_mid = (ax_y0 + ax_y1) // 2

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',

        # X axis
        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x1}" y2="{ax_y0}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="{AXIS_W}"/>',
        f'<polygon points="{arrow_x}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',

        # Y axis
        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y1}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="{AXIS_W}"/>',
        f'<polygon points="{arrow_y}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',

        # Axis labels
        f'<text x="{cx_mid}" y="{ax_y0 + 56}" '
        f'text-anchor="middle" font-size="{LABEL_FS}" '
        f'font-family="ui-sans-serif,sans-serif" letter-spacing="0.04em" '
        f'fill="{LINE_COLOR}" fill-opacity="0.75">{pc1_label}</text>',

        f'<text x="{ax_x0 - 62}" y="{cy_mid}" '
        f'text-anchor="middle" font-size="{LABEL_FS}" '
        f'font-family="ui-sans-serif,sans-serif" letter-spacing="0.04em" '
        f'fill="{LINE_COLOR}" fill-opacity="0.75" '
        f'transform="rotate(-90 {ax_x0 - 62} {cy_mid})">{pc2_label}</text>',
    ]

    for t in range(N_TICKS + 1):
        frac = t / N_TICKS

        tx      = ax_x0 + frac * (ax_x1 - ax_x0)
        x_score = x_min + frac * x_range
        is_zero = abs(x_score) < x_range / (N_TICKS * 2)
        t_op    = AXIS_OP if is_zero else AXIS_OP * 0.55
        t_w     = 2.0    if is_zero else 1.5
        lines.append(
            f'<line x1="{tx:.1f}" y1="{ax_y0 - TICK_LEN}" '
            f'x2="{tx:.1f}" y2="{ax_y0 + TICK_LEN}" '
            f'stroke="{LINE_COLOR}" stroke-opacity="{t_op:.2f}" stroke-width="{t_w}"/>'
        )
        lines.append(
            f'<text x="{tx:.1f}" y="{ax_y0 + TICK_LEN + 28:.1f}" '
            f'text-anchor="middle" font-size="{TICK_FS}" '
            f'font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="{t_op:.2f}">'
            f'{x_score:.1f}</text>'
        )

        ty      = ax_y0 + frac * (ax_y1 - ax_y0)
        y_score = y_min + frac * y_range
        is_zero = abs(y_score) < y_range / (N_TICKS * 2)
        t_op    = AXIS_OP if is_zero else AXIS_OP * 0.55
        t_w     = 2.0    if is_zero else 1.5
        lines.append(
            f'<line x1="{ax_x0 - TICK_LEN}" y1="{ty:.1f}" '
            f'x2="{ax_x0 + TICK_LEN}" y2="{ty:.1f}" '
            f'stroke="{LINE_COLOR}" stroke-opacity="{t_op:.2f}" stroke-width="{t_w}"/>'
        )
        lines.append(
            f'<text x="{ax_x0 - TICK_LEN - 10:.1f}" y="{ty + TICK_FS * 0.35:.1f}" '
            f'text-anchor="end" font-size="{TICK_FS}" '
            f'font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="{t_op:.2f}">'
            f'{y_score:.1f}</text>'
        )

    for i, stem in enumerate(df["stem"]):
        px, py = to_px(x_vals[i], y_vals[i])
        glyph  = _glyph_svg(stem)
        if glyph:
            href, nat = glyph
            lines.append(_scaled_image_tag(href, px, py, GLYPH, GLYPH, nat))
        else:
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" '
                f'fill="{LINE_COLOR}" fill-opacity="0.4"/>'
            )

    lines.append('</svg>')
    out = PLOT_DIR / "pca_full.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Full PCA plot saved to {out}")


# ── Sparse PCA grid — one tag per grid cell, no overlap ───────────────────────

def plot_pca_sparse(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    explained: np.ndarray,
    family_labels: np.ndarray,
    palette: dict[int, str] | None = None,
    names: dict[int, str] | None = None,
    grid_cols: int | None = None,
    glyph_size: int = 150,
):
    """Sparse PCA: PCA space divided into a regular grid, one tag shown per cell.

    For each grid cell that contains at least one tag, the tag nearest to that
    cell's centre is displayed. This eliminates overlap while preserving the
    large-scale structure of the PCA space. Glyphs are sized to fill their cell.

    Output: data/plots/pca_sparse.svg
    """
    if palette is None:
        palette = make_palette(int(family_labels.max()))
    if names is None:
        names = make_names(int(family_labels.max()))

    stems  = list(df["stem"])
    n      = len(stems)
    x_vals = X_pca[:, 0]
    y_vals = X_pca[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    # Auto grid: aim for ~1.5× more cells than tags so space breathes
    if grid_cols is None:
        grid_cols = max(6, int(np.ceil(np.sqrt(n * 1.5))))
    grid_rows = max(4, int(np.ceil(grid_cols * y_range / x_range)))

    # ── Assign each tag to its grid cell ──────────────────────────────────────
    col_idx = np.clip(
        ((x_vals - x_min) / x_range * grid_cols).astype(int), 0, grid_cols - 1)
    row_idx = np.clip(
        ((y_max - y_vals) / y_range * grid_rows).astype(int), 0, grid_rows - 1)

    # For each occupied cell, keep the tag closest to that cell's PCA centre
    cell_tags: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (int(row_idx[i]), int(col_idx[i]))
        cell_tags.setdefault(key, []).append(i)

    def cell_pca_centre(row: int, col: int) -> tuple[float, float]:
        cx = x_min + (col + 0.5) / grid_cols * x_range
        cy = y_max - (row + 0.5) / grid_rows * y_range
        return cx, cy

    selected: dict[tuple, int] = {}   # (row, col) → tag index
    for (row, col), indices in cell_tags.items():
        cx, cy = cell_pca_centre(row, col)
        dists  = [(x_vals[i] - cx) ** 2 + (y_vals[i] - cy) ** 2 for i in indices]
        selected[(row, col)] = indices[int(np.argmin(dists))]

    # ── Layout ─────────────────────────────────────────────────────────────────
    PADDING    = 12          # gap between glyphs
    CELL_W     = glyph_size + PADDING
    CELL_H     = glyph_size + PADDING
    MARGIN_L   = 80          # left margin (for Y-axis label)
    MARGIN_T   = 40
    MARGIN_R   = 200         # right margin (legend)
    MARGIN_B   = 180         # bottom margin (interpretation panel)
    HALO_R     = glyph_size // 2 + 6   # family halo slightly larger than glyph

    W = MARGIN_L + grid_cols * CELL_W + MARGIN_R
    H = MARGIN_T + grid_rows * CELL_H + MARGIN_B

    scatter_w = grid_cols * CELL_W
    scatter_h = grid_rows * CELL_H

    LINE_COLOR = "#000000"
    AXIS_OP    = 0.55
    LABEL_FS   = 26
    TICK_FS    = 17

    def cell_to_px(row: int, col: int) -> tuple[float, float]:
        cx = MARGIN_L + (col + 0.5) * CELL_W
        cy = MARGIN_T + (row + 0.5) * CELL_H
        return cx, cy

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    ]

    # ── Family halos (behind glyphs) ───────────────────────────────────────────
    for (row, col), idx in selected.items():
        cx, cy = cell_to_px(row, col)
        fam    = int(family_labels[idx])
        color  = palette.get(fam, "#888888")
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{HALO_R}" '
            f'fill="{color}" fill-opacity="0.18"/>'
        )

    # ── Axes (drawn over halos, behind glyphs) ─────────────────────────────────
    ax_x0 = MARGIN_L
    ax_x1 = MARGIN_L + scatter_w
    ax_y0 = MARGIN_T + scatter_h   # bottom
    ax_y1 = MARGIN_T               # top
    cx_mid = (ax_x0 + ax_x1) // 2
    cy_mid = (ax_y0 + ax_y1) // 2
    ARROW = 11

    lines += [
        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x1}" y2="{ax_y0}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="2"/>',
        f'<polygon points="{ax_x1},{ax_y0} {ax_x1-ARROW},{ax_y0-ARROW//2} '
        f'{ax_x1-ARROW},{ax_y0+ARROW//2}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',
        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y1}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="2"/>',
        f'<polygon points="{ax_x0},{ax_y1} {ax_x0-ARROW//2},{ax_y1+ARROW} '
        f'{ax_x0+ARROW//2},{ax_y1+ARROW}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',
        # Axis labels
        f'<text x="{cx_mid}" y="{ax_y0 + 42}" text-anchor="middle" '
        f'font-size="{LABEL_FS}" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.60">'
        f'PC1 {explained[0]:.1f}%</text>',
        f'<text x="{ax_x0 - 54}" y="{cy_mid}" text-anchor="middle" '
        f'font-size="{LABEL_FS}" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.60" '
        f'transform="rotate(-90 {ax_x0 - 54} {cy_mid})">'
        f'PC2 {explained[1]:.1f}%</text>',
    ]

    # Tick labels at grid cell boundaries (showing PCA score)
    N_XTICKS = min(6, grid_cols)
    N_YTICKS = min(5, grid_rows)
    for t in range(N_XTICKS + 1):
        frac  = t / N_XTICKS
        tx    = ax_x0 + frac * scatter_w
        score = x_min + frac * x_range
        lines.append(
            f'<text x="{tx:.1f}" y="{ax_y0 + 22:.1f}" text-anchor="middle" '
            f'font-size="{TICK_FS}" font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="0.35">{score:.1f}</text>'
        )
    for t in range(N_YTICKS + 1):
        frac  = t / N_YTICKS
        ty    = ax_y1 + frac * scatter_h
        score = y_max - frac * y_range
        lines.append(
            f'<text x="{ax_x0 - 8:.1f}" y="{ty + TICK_FS * 0.35:.1f}" '
            f'text-anchor="end" '
            f'font-size="{TICK_FS}" font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="0.35">{score:.1f}</text>'
        )

    # ── Tag glyphs ─────────────────────────────────────────────────────────────
    for (row, col), idx in selected.items():
        cx, cy = cell_to_px(row, col)
        stem   = stems[idx]
        glyph  = _glyph_svg(stem)
        if glyph:
            href, nat = glyph
            lines.append(_scaled_image_tag(href, cx, cy, glyph_size, glyph_size, nat))
        else:
            fam   = int(family_labels[idx])
            color = palette.get(fam, "#000000")
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                f'fill="{color}" fill-opacity="0.5"/>'
            )

    # ── Legend (top right) ─────────────────────────────────────────────────────
    leg_x = ax_x1 + 24
    leg_y = MARGIN_T + 20
    DOT_R = 8
    for fam, name in names.items():
        color = palette.get(fam, "#888")
        lines.append(
            f'<circle cx="{leg_x + DOT_R:.0f}" cy="{leg_y:.0f}" r="{DOT_R}" '
            f'fill="{color}" fill-opacity="0.80"/>'
        )
        lines.append(
            f'<text x="{leg_x + DOT_R * 2 + 6:.0f}" y="{leg_y + 5:.0f}" '
            f'font-size="18" font-family="ui-sans-serif,sans-serif" '
            f'fill="{LINE_COLOR}" fill-opacity="0.55">{name}</text>'
        )
        leg_y += 28

    # ── Stats note ─────────────────────────────────────────────────────────────
    n_shown = len(selected)
    lines.append(
        f'<text x="{leg_x}" y="{leg_y + 20}" '
        f'font-size="15" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.35">'
        f'{n_shown} of {n} shown</text>'
    )
    lines.append(
        f'<text x="{leg_x}" y="{leg_y + 38}" '
        f'font-size="15" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.35">'
        f'grid {grid_cols}×{grid_rows}</text>'
    )

    # ── Interpretation panel ───────────────────────────────────────────────────
    interp_y0 = MARGIN_T + scatter_h + 60
    lines += [
        f'<line x1="{ax_x0}" y1="{interp_y0 - 18}" x2="{ax_x1}" y2="{interp_y0 - 18}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="0.10" stroke-width="1"/>',
        f'<text x="{ax_x0}" y="{interp_y0}" text-anchor="start" '
        f'font-size="21" font-family="ui-sans-serif,sans-serif" font-weight="600" '
        f'fill="{LINE_COLOR}" fill-opacity="0.75">'
        f'PC1 — {explained[0]:.1f}% of variation</text>',
        f'<text x="{(ax_x0 + ax_x1) // 2 + 20}" y="{interp_y0}" text-anchor="start" '
        f'font-size="21" font-family="ui-sans-serif,sans-serif" font-weight="600" '
        f'fill="{LINE_COLOR}" fill-opacity="0.75">'
        f'PC2 — {explained[1]:.1f}% of variation</text>',
        # Note about sampling
        f'<text x="{ax_x1}" y="{interp_y0 + 28}" text-anchor="end" '
        f'font-size="15" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.30">'
        f'one representative per grid cell · nearest to cell centre</text>',
    ]

    lines.append('</svg>')
    out = PLOT_DIR / "pca_sparse.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    n_empty = grid_cols * grid_rows - n_shown
    print(f"  ✓ Sparse PCA saved to {out}  "
          f"[{grid_cols}×{grid_rows} grid, {n_shown} tags shown, {n_empty} empty cells]")


# ── Preview PCA — landing page ─────────────────────────────────────────────────

def plot_pca_preview(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    explained: np.ndarray,
    family_labels: np.ndarray,
    loadings: "pd.DataFrame",
    palette: dict[int, str] | None = None,
    names: dict[int, str] | None = None,
):
    """Landing-page PCA preview: larger glyphs, family colour halos, brief interpretation.

    Family membership shown as a soft coloured circle behind each tag glyph.
    Interpretation panel below the scatter summarises what PC1 and PC2 mean.
    Output: data/plots/pca_preview.svg
    """
    if palette is None:
        palette = make_palette(int(family_labels.max()))
    if names is None:
        names = make_names(int(family_labels.max()))

    # ── Canvas ─────────────────────────────────────────────────────────────────
    W          = 1400
    MARGIN     = 90        # around scatter area
    GLYPH      = 220       # glyph bounding box (larger than full view)
    HALO_R     = 120       # family accent circle radius
    SCATTER_H  = 900       # height of scatter region
    INTERP_H   = 220       # interpretation text panel height
    LINK_H     = 52        # "view full" link area
    H          = SCATTER_H + INTERP_H + LINK_H

    x_vals = X_pca[:, 0]
    y_vals = X_pca[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    def to_px(xi, yi):
        px = MARGIN + (xi - x_min) / x_range * (W - 2 * MARGIN)
        py = MARGIN + (y_max - yi) / y_range * (SCATTER_H - 2 * MARGIN)
        return px, py

    LINE_COLOR = "#000000"
    AXIS_OP    = 0.60
    AXIS_W     = 2.0
    TICK_LEN   = 10
    ARROW      = 12
    LABEL_FS   = 28
    TICK_FS    = 18
    N_TICKS    = 4

    pc1_label = f"PC1  {explained[0]:.1f}%"
    pc2_label = f"PC2  {explained[1]:.1f}%"

    ax_x0, ax_x1 = MARGIN, W - MARGIN
    ax_y0, ax_y1 = SCATTER_H - MARGIN, MARGIN    # y0=bottom, y1=top

    arrow_x = (f"{ax_x1},{ax_y0} "
               f"{ax_x1-ARROW},{ax_y0-ARROW//2} "
               f"{ax_x1-ARROW},{ax_y0+ARROW//2}")
    arrow_y = (f"{ax_x0},{ax_y1} "
               f"{ax_x0-ARROW//2},{ax_y1+ARROW} "
               f"{ax_x0+ARROW//2},{ax_y1+ARROW}")

    cx_mid = (ax_x0 + ax_x1) // 2
    cy_mid = (ax_y0 + ax_y1) // 2

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    ]

    # ── Family halos (drawn before glyphs so tags sit on top) ──────────────────
    stems = list(df["stem"])
    for i, stem in enumerate(stems):
        px, py = to_px(x_vals[i], y_vals[i])
        fam    = int(family_labels[i])
        color  = palette.get(fam, "#888888")
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{HALO_R}" '
            f'fill="{color}" fill-opacity="0.10"/>'
        )

    # ── Axes ───────────────────────────────────────────────────────────────────
    lines += [
        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x1}" y2="{ax_y0}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="{AXIS_W}"/>',
        f'<polygon points="{arrow_x}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',

        f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y1}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="{AXIS_OP}" stroke-width="{AXIS_W}"/>',
        f'<polygon points="{arrow_y}" fill="{LINE_COLOR}" fill-opacity="{AXIS_OP}"/>',

        f'<text x="{cx_mid}" y="{ax_y0 + 44}" '
        f'text-anchor="middle" font-size="{LABEL_FS}" '
        f'font-family="ui-sans-serif,sans-serif" letter-spacing="0.05em" '
        f'fill="{LINE_COLOR}" fill-opacity="0.65">{pc1_label}</text>',

        f'<text x="{ax_x0 - 52}" y="{cy_mid}" '
        f'text-anchor="middle" font-size="{LABEL_FS}" '
        f'font-family="ui-sans-serif,sans-serif" letter-spacing="0.05em" '
        f'fill="{LINE_COLOR}" fill-opacity="0.65" '
        f'transform="rotate(-90 {ax_x0 - 52} {cy_mid})">{pc2_label}</text>',
    ]

    # Tick marks
    for t in range(N_TICKS + 1):
        frac = t / N_TICKS

        tx      = ax_x0 + frac * (ax_x1 - ax_x0)
        x_score = x_min + frac * x_range
        is_zero = abs(x_score) < x_range / (N_TICKS * 2)
        t_op    = AXIS_OP if is_zero else AXIS_OP * 0.55
        lines.append(
            f'<line x1="{tx:.1f}" y1="{ax_y0 - TICK_LEN}" '
            f'x2="{tx:.1f}" y2="{ax_y0 + TICK_LEN}" '
            f'stroke="{LINE_COLOR}" stroke-opacity="{t_op:.2f}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{tx:.1f}" y="{ax_y0 + TICK_LEN + 22:.1f}" '
            f'text-anchor="middle" font-size="{TICK_FS}" '
            f'font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="{t_op:.2f}">'
            f'{x_score:.1f}</text>'
        )

        ty      = ax_y0 + frac * (ax_y1 - ax_y0)
        y_score = y_min + frac * y_range
        is_zero = abs(y_score) < y_range / (N_TICKS * 2)
        t_op    = AXIS_OP if is_zero else AXIS_OP * 0.55
        lines.append(
            f'<line x1="{ax_x0 - TICK_LEN}" y1="{ty:.1f}" '
            f'x2="{ax_x0 + TICK_LEN}" y2="{ty:.1f}" '
            f'stroke="{LINE_COLOR}" stroke-opacity="{t_op:.2f}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{ax_x0 - TICK_LEN - 8:.1f}" y="{ty + TICK_FS * 0.35:.1f}" '
            f'text-anchor="end" font-size="{TICK_FS}" '
            f'font-family="ui-monospace,monospace" '
            f'fill="{LINE_COLOR}" fill-opacity="{t_op:.2f}">'
            f'{y_score:.1f}</text>'
        )

    # ── Tag glyphs ─────────────────────────────────────────────────────────────
    for i, stem in enumerate(stems):
        px, py = to_px(x_vals[i], y_vals[i])
        glyph  = _glyph_svg(stem)
        if glyph:
            href, nat = glyph
            lines.append(_scaled_image_tag(href, px, py, GLYPH, GLYPH, nat))
        else:
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" '
                f'fill="{LINE_COLOR}" fill-opacity="0.35"/>'
            )

    # ── Family legend (top-right corner) ───────────────────────────────────────
    leg_x = W - MARGIN - 10
    leg_y = MARGIN - 10
    DOT_R = 9
    for fam, name in names.items():
        color = palette.get(fam, "#888")
        lines.append(
            f'<circle cx="{leg_x - 110:.0f}" cy="{leg_y:.0f}" r="{DOT_R}" '
            f'fill="{color}" fill-opacity="0.75"/>'
        )
        lines.append(
            f'<text x="{leg_x - 96:.0f}" y="{leg_y + 5:.0f}" '
            f'font-size="18" font-family="ui-sans-serif,sans-serif" '
            f'fill="{LINE_COLOR}" fill-opacity="0.55">{name}</text>'
        )
        leg_y += 28

    # ── Interpretation panel ───────────────────────────────────────────────────
    if "PC1" in loadings.columns and "PC2" in loadings.columns:
        pc1_pos, pc1_neg = _pc_brief_labels(loadings, "PC1", n=2)
        pc2_pos, pc2_neg = _pc_brief_labels(loadings, "PC2", n=2)
    else:
        pc1_pos = pc1_neg = pc2_pos = pc2_neg = []

    interp_y0 = SCATTER_H + 38
    TEXT_FS   = 23
    LABEL_FS2 = 20
    half_W    = W // 2

    def interp_text(x, y, anchor, fs, opacity, content, bold=False):
        weight = ' font-weight="600"' if bold else ''
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                f'font-size="{fs}" font-family="ui-sans-serif,sans-serif" '
                f'fill="{LINE_COLOR}" fill-opacity="{opacity}"{weight}>'
                f'{content}</text>')

    # PC1 block (left half)
    pc1_pct = f"{explained[0]:.1f}%"
    lines.append(interp_text(MARGIN, interp_y0, "start", TEXT_FS, 0.80,
                              f"PC1 — {pc1_pct} of variation", bold=True))
    if pc1_pos:
        lines.append(interp_text(MARGIN, interp_y0 + 34, "start", LABEL_FS2, 0.50,
                                  f"→ high:  {' · '.join(pc1_pos)}"))
    if pc1_neg:
        lines.append(interp_text(MARGIN, interp_y0 + 62, "start", LABEL_FS2, 0.50,
                                  f"← low:   {' · '.join(pc1_neg)}"))
    lines.append(interp_text(MARGIN, interp_y0 + 96, "start", LABEL_FS2, 0.40,
                              "complexity ↔ simplicity of stroke structure"))

    # PC2 block (right half)
    pc2_pct = f"{explained[1]:.1f}%"
    lines.append(interp_text(half_W + 20, interp_y0, "start", TEXT_FS, 0.80,
                              f"PC2 — {pc2_pct} of variation", bold=True))
    if pc2_pos:
        lines.append(interp_text(half_W + 20, interp_y0 + 34, "start", LABEL_FS2, 0.50,
                                  f"↑ high:  {' · '.join(pc2_pos)}"))
    if pc2_neg:
        lines.append(interp_text(half_W + 20, interp_y0 + 62, "start", LABEL_FS2, 0.50,
                                  f"↓ low:   {' · '.join(pc2_neg)}"))
    lines.append(interp_text(half_W + 20, interp_y0 + 96, "start", LABEL_FS2, 0.40,
                              "form ↔ proportions and fill"))

    # Dividing line between panels
    div_y = interp_y0 - 14
    lines.append(
        f'<line x1="{MARGIN}" y1="{div_y}" x2="{W - MARGIN}" y2="{div_y}" '
        f'stroke="{LINE_COLOR}" stroke-opacity="0.12" stroke-width="1"/>'
    )

    # ── "View full" link ────────────────────────────────────────────────────────
    link_y = SCATTER_H + INTERP_H + 30
    lines.append(
        f'<a href="pca_full.svg" target="_blank">'
        f'<text x="{W - MARGIN}" y="{link_y}" text-anchor="end" '
        f'font-size="22" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.40" '
        f'text-decoration="underline">View full PCA →</text>'
        f'</a>'
    )

    lines.append('</svg>')
    out = PLOT_DIR / "pca_preview.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ PCA preview saved to {out}")


# ── Full dendrogram — all tags, portrait ──────────────────────────────────────

def plot_dendrogram_full(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    Z: np.ndarray,
    family_labels: np.ndarray,
    palette: dict[int, str] | None = None,
):
    """Full hierarchical clustering dendrogram, portrait orientation.

    Root at top, leaves spread horizontally at the bottom.
    Glyph row below the tree, horizontal family colour strip below glyphs.
    Output: data/plots/dendrogram_full.svg
    """
    if palette is None:
        palette = make_palette(int(family_labels.max()))

    stems = list(df["stem"])
    n     = len(stems)
    d_max = float(max(Z[:, 2]))

    ddata = dendrogram(Z, no_plot=True, labels=stems)

    stem_to_family = {stem: int(family_labels[i]) for i, stem in enumerate(stems)}

    FAMILY_COLORS = palette

    # Layout
    MARGIN_T  = 60
    MARGIN_B  = 48
    MARGIN_L  = 60
    MARGIN_R  = 60
    TREE_H    = 900
    GAP       = 20
    GLYPH_H   = 90
    STRIP_GAP = 14
    STRIP_H   = 10
    LINE_COLOR = "#000000"

    col_w = max(110, (900 - MARGIN_L - MARGIN_R) // max(n, 1))
    W     = max(900, n * col_w + MARGIN_L + MARGIN_R)

    tree_top_y  = MARGIN_T
    tree_bot_y  = MARGIN_T + TREE_H
    glyph_cy    = tree_bot_y + GAP + GLYPH_H / 2
    strip_top_y = tree_bot_y + GAP + GLYPH_H + STRIP_GAP
    H           = int(strip_top_y + STRIP_H + MARGIN_B)

    x_left  = MARGIN_L
    x_right = W - MARGIN_R
    i_min   = 5
    i_span  = max((n - 1) * 10, 1)

    def svgx(i_val: float) -> float:
        return x_left + (i_val - i_min) / i_span * (x_right - x_left)

    def svgy(d_val: float) -> float:
        return tree_top_y + (1.0 - d_val / d_max) * TREE_H

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    ]

    for icoord_link, dcoord_link in zip(ddata["icoord"], ddata["dcoord"]):
        d_merge = dcoord_link[1]
        opacity = 0.25 + 0.75 * (1.0 - d_merge / d_max)
        pts = [(svgx(ic), svgy(d))
               for ic, d in zip(icoord_link, dcoord_link)]
        path_d = (f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
                  f"L {pts[1][0]:.1f},{pts[1][1]:.1f} "
                  f"L {pts[2][0]:.1f},{pts[2][1]:.1f} "
                  f"L {pts[3][0]:.1f},{pts[3][1]:.1f}")
        lines.append(
            f'<path d="{path_d}" stroke="{LINE_COLOR}" stroke-opacity="{opacity:.2f}" '
            f'stroke-width="1.5" fill="none" stroke-linejoin="round"/>'
        )

    leaf_order = ddata["ivl"]
    half_col   = (x_right - x_left) / max(n - 1, 1) / 2

    for rank, stem in enumerate(leaf_order):
        cx = svgx(5 + rank * 10)
        gw = max(60, col_w - 8)

        glyph = _glyph_svg(stem)
        if glyph:
            href, nat = glyph
            lines.append(_scaled_image_tag(href, cx, glyph_cy, gw, GLYPH_H, nat))
        else:
            lines.append(f'<circle cx="{cx:.1f}" cy="{glyph_cy:.1f}" r="4" '
                         f'fill="{LINE_COLOR}" fill-opacity="0.4"/>')

    # Horizontal family strip
    if leaf_order:
        run_start  = 0
        run_family = stem_to_family.get(leaf_order[0], 1)

        def _flush_strip(start_rank: int, end_rank: int, family: int):
            color = FAMILY_COLORS[family]
            rx    = svgx(5 + start_rank * 10) - half_col
            rw    = svgx(5 + end_rank   * 10) + half_col - rx
            lines.append(
                f'<rect x="{rx:.1f}" y="{strip_top_y:.1f}" '
                f'width="{rw:.1f}" height="{STRIP_H}" '
                f'fill="{color}" rx="2"/>'
            )

        for rank, stem in enumerate(leaf_order):
            fam = stem_to_family.get(stem, 1)
            if fam != run_family:
                _flush_strip(run_start, rank - 1, run_family)
                run_start  = rank
                run_family = fam
        _flush_strip(run_start, len(leaf_order) - 1, run_family)

    lines.append('</svg>')

    out = PLOT_DIR / "dendrogram_full.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Full dendrogram saved to {out}  [{W}×{H}px, portrait]")


# ── Preview dendrogram — 3 reps per family ────────────────────────────────────

def plot_dendrogram_preview(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    Z: np.ndarray,
    family_labels: np.ndarray,
    palette: dict[int, str] | None = None,
    names: dict[int, str] | None = None,
    n_reps: int = 3,
):
    """Actual dendrogram above k family columns.

    Top section: the real Ward linkage tree (all leaves, data-driven) scaled
    to fit the canvas width, with a family colour strip at its base.
    Bottom section: k family columns (one per family, in dendrogram leaf order)
    each showing n_reps representative tag glyphs (closest to family centroid).
    Output: data/plots/dendrogram_preview.svg
    """
    k = int(family_labels.max())
    if palette is None:
        palette = make_palette(k)
    if names is None:
        names = make_names(k)

    stems  = list(df["stem"])
    n      = len(stems)
    d_max  = float(max(Z[:, 2]))

    ddata          = dendrogram(Z, no_plot=True, labels=stems)
    leaf_order     = ddata["ivl"]
    stem_to_family = {stem: int(family_labels[i]) for i, stem in enumerate(stems)}

    FAMILY_COLORS = palette

    # ── n_reps representatives per family (closest to centroid) ──────────────
    family_members: dict[int, list[int]] = {f: [] for f in range(1, k + 1)}
    for i, fam in enumerate(family_labels):
        family_members[int(fam)].append(i)

    family_reps: dict[int, list[str]] = {}
    for fam, indices in family_members.items():
        if not indices:
            family_reps[fam] = []
            continue
        fam_X    = X_scaled[indices]
        centroid = fam_X.mean(axis=0)
        dists    = np.linalg.norm(fam_X - centroid, axis=1)
        n_pick   = min(n_reps, len(indices))
        family_reps[fam] = [stems[indices[j]] for j in np.argsort(dists)[:n_pick]]

    # ── Column order follows dendrogram leaf order ─────────────────────────────
    family_run_order: list[int] = []
    for stem in leaf_order:
        fam = stem_to_family.get(stem, 1)
        if not family_run_order or family_run_order[-1] != fam:
            family_run_order.append(fam)
    for fam in range(1, k + 1):
        if fam not in family_run_order:
            family_run_order.append(fam)
    family_run_order = family_run_order[:k]

    # ── Dimensions — full-page width, columns and tree share the same span ───────
    W          = 1400          # wide canvas — fills page at normal zoom
    MARGIN_T   = 40
    MARGIN_L   = 30
    MARGIN_R   = 30
    MARGIN_B   = 70
    LINE_COLOR = "#000000"

    # Tree section
    TREE_H      = 300
    STRIP_H     = 10
    STRIP_GAP   = 14
    tree_top_y  = MARGIN_T
    tree_bot_y  = MARGIN_T + TREE_H
    strip_top_y = tree_bot_y + 10
    col_top_y   = strip_top_y + STRIP_H + STRIP_GAP

    # Column section — COL_W derived from W so columns exactly match tree width
    N_REPS    = n_reps
    COL_GAP   = 20
    HEADER_H  = 52
    GLYPH_SZ  = 160
    GLYPH_GAP = 14

    avail_w = W - MARGIN_L - MARGIN_R
    COL_W   = (avail_w - (k - 1) * COL_GAP) // k

    H = col_top_y + HEADER_H + N_REPS * (GLYPH_SZ + GLYPH_GAP) + MARGIN_B

    # Dendrogram coordinate mapping (tree spans full W)
    x_left  = MARGIN_L
    x_right = W - MARGIN_R
    i_min   = 5
    i_span  = max((n - 1) * 10, 1)
    half_col = (x_right - x_left) / max(n - 1, 1) / 2

    def svgx(i_val: float) -> float:
        return x_left + (i_val - i_min) / i_span * (x_right - x_left)

    def svgy(d_val: float) -> float:
        return tree_top_y + (1.0 - d_val / d_max) * TREE_H

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    ]

    # ── Actual dendrogram tree ─────────────────────────────────────────────────
    for icoord_link, dcoord_link in zip(ddata["icoord"], ddata["dcoord"]):
        d_merge = dcoord_link[1]
        opacity = 0.25 + 0.75 * (1.0 - d_merge / d_max)
        pts = [(svgx(ic), svgy(d))
               for ic, d in zip(icoord_link, dcoord_link)]
        path_d = (f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
                  f"L {pts[1][0]:.1f},{pts[1][1]:.1f} "
                  f"L {pts[2][0]:.1f},{pts[2][1]:.1f} "
                  f"L {pts[3][0]:.1f},{pts[3][1]:.1f}")
        lines.append(
            f'<path d="{path_d}" stroke="{LINE_COLOR}" stroke-opacity="{opacity:.2f}" '
            f'stroke-width="1.5" fill="none" stroke-linejoin="round"/>'
        )

    # ── Family colour strip (joins tree to columns) ────────────────────────────
    if leaf_order:
        run_start  = 0
        run_family = stem_to_family.get(leaf_order[0], 1)

        def _flush_strip(start_rank: int, end_rank: int, family: int):
            color = FAMILY_COLORS[family]
            rx    = svgx(5 + start_rank * 10) - half_col
            rw    = svgx(5 + end_rank   * 10) + half_col - rx
            lines.append(
                f'<rect x="{rx:.1f}" y="{strip_top_y:.1f}" '
                f'width="{rw:.1f}" height="{STRIP_H}" '
                f'fill="{color}" rx="2"/>'
            )

        for rank, stem in enumerate(leaf_order):
            fam = stem_to_family.get(stem, 1)
            if fam != run_family:
                _flush_strip(run_start, rank - 1, run_family)
                run_start  = rank
                run_family = fam
        _flush_strip(run_start, len(leaf_order) - 1, run_family)

    # ── 3 family columns ───────────────────────────────────────────────────────
    for col_idx, fam in enumerate(family_run_order):
        col_x  = MARGIN_L + col_idx * (COL_W + COL_GAP)
        col_cx = col_x + COL_W / 2
        color  = FAMILY_COLORS.get(fam, "#888")
        reps   = family_reps.get(fam, [])

        # Header
        lines.append(
            f'<rect x="{col_x}" y="{col_top_y}" '
            f'width="{COL_W}" height="{HEADER_H}" '
            f'fill="{color}" fill-opacity="0.85" rx="4"/>'
        )
        label_y = col_top_y + HEADER_H * 0.62
        lines.append(
            f'<text x="{col_cx:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" font-size="22" '
            f'font-family="ui-sans-serif,sans-serif" font-weight="600" '
            f'fill="#ffffff" fill-opacity="0.92" letter-spacing="0.06em">'
            f'{names.get(fam, f"Family {fam}")}</text>'
        )

        # Representative glyphs
        for rep_idx in range(N_REPS):
            glyph_y = col_top_y + HEADER_H + rep_idx * (GLYPH_SZ + GLYPH_GAP) + GLYPH_GAP // 2
            cy_g    = glyph_y + GLYPH_SZ / 2

            if rep_idx < len(reps):
                stem  = reps[rep_idx]
                glyph = _glyph_svg(stem)
                if glyph:
                    href, nat = glyph
                    lines.append(
                        f'<rect x="{col_x + 8}" y="{glyph_y:.1f}" '
                        f'width="{COL_W - 16}" height="{GLYPH_SZ}" '
                        f'fill="{color}" fill-opacity="0.05" rx="3"/>'
                    )
                    lines.append(
                        _scaled_image_tag(href, col_cx, cy_g,
                                          COL_W - 24, GLYPH_SZ - 16, nat)
                    )
                else:
                    lines.append(
                        f'<circle cx="{col_cx:.1f}" cy="{cy_g:.1f}" r="8" '
                        f'fill="{LINE_COLOR}" fill-opacity="0.25"/>'
                    )
            else:
                lines.append(
                    f'<rect x="{col_x + 8}" y="{glyph_y:.1f}" '
                    f'width="{COL_W - 16}" height="{GLYPH_SZ}" '
                    f'fill="{LINE_COLOR}" fill-opacity="0.03" rx="3"/>'
                )

        # Bottom border
        col_bot = col_top_y + HEADER_H + N_REPS * (GLYPH_SZ + GLYPH_GAP)
        lines.append(
            f'<line x1="{col_x}" y1="{col_bot:.1f}" '
            f'x2="{col_x + COL_W}" y2="{col_bot:.1f}" '
            f'stroke="{LINE_COLOR}" stroke-opacity="0.08" stroke-width="1"/>'
        )

    # ── "View full" link ────────────────────────────────────────────────────────
    link_y = H - 22
    lines.append(
        f'<a href="dendrogram_full.svg" target="_blank">'
        f'<text x="{W - MARGIN_R}" y="{link_y}" text-anchor="end" '
        f'font-size="20" font-family="ui-sans-serif,sans-serif" '
        f'fill="{LINE_COLOR}" fill-opacity="0.38" '
        f'text-decoration="underline">View full dendrogram →</text>'
        f'</a>'
    )

    lines.append('</svg>')
    out = PLOT_DIR / "dendrogram_preview.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Dendrogram preview saved to {out}  [{W}×{H}px]")


# ── PCA interpretation document ───────────────────────────────────────────────

def _write_pca_interpretation(
    loadings: "pd.DataFrame",
    explained: np.ndarray,
    feature_desc: dict,
    n_top: int = 5,
) -> None:
    """Write a structured markdown interpretation of PC1 and PC2 loadings.

    Saves to <project>/data/plots/pca_interpretation.md.
    """
    OUT_INTERP = PLOT_DIR / "pca_interpretation.md"
    OUT_INTERP.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# other.codes — PCA Interpretation",
        "",
        "Principal Component Analysis reduces the 17 morphological measurements "
        "to a small number of axes that capture the major ways tags differ from one another. "
        "This document describes what each axis represents.",
        "",
        f"**Variance explained:** PC1 = {explained[0]:.1f}%,  "
        f"PC2 = {explained[1]:.1f}%,  "
        f"total (first 2 PCs) = {sum(explained[:2]):.1f}%",
        "",
        "---",
        "",
    ]

    for pc_name in [c for c in loadings.columns if c in ("PC1", "PC2")]:
        idx = int(pc_name[2]) - 1
        col = loadings[pc_name].sort_values(ascending=False)
        top_pos = col.head(n_top)
        top_neg = col.tail(n_top).iloc[::-1]

        lines += [
            f"## {pc_name} — {explained[idx]:.1f}% of variance",
            "",
            f"### Features that drive {pc_name} higher (positive loadings)",
            "",
        ]
        for feat, val in top_pos.items():
            desc = feature_desc.get(feat, "")
            lines.append(f"- **{feat}** ({val:+.3f}) — {desc}")

        lines += [
            "",
            f"### Features that drive {pc_name} lower (negative loadings)",
            "",
        ]
        for feat, val in top_neg.items():
            desc = feature_desc.get(feat, "")
            lines.append(f"- **{feat}** ({val:+.3f}) — {desc}")

        if pc_name == "PC1":
            lines += [
                "",
                "### Intuitive interpretation of PC1",
                "",
                "PC1 separates tags by **structural complexity and stroke variability**. "
                "Tags at the high end tend to have more branching, more enclosed loops, "
                "highly variable stroke widths, and complex vector paths — "
                "suggesting an elaborate, multi-element style with thick and thin contrast. "
                "Tags at the low end are simpler: fewer branches, thinner and more uniform "
                "strokes, and fewer closed shapes — suggesting a leaner, more linear hand.",
                "",
            ]
        elif pc_name == "PC2":
            lines += [
                "",
                "### Intuitive interpretation of PC2",
                "",
                "PC2 separates tags by **overall form and proportions**. "
                "Tags at the high end tend to be wider relative to their height, "
                "with a more compact and solid filled form. "
                "Tags at the low end are taller, more elongated, and more eccentrically shaped "
                "— suggesting upright, vertical letterforms vs squat, horizontal ones. "
                "Skeleton-to-area and fill density also load here, "
                "reflecting how much of the bounding box is actually ink.",
                "",
            ]

    lines += [
        "---",
        "",
        "*Generated automatically by `pipeline/cluster.py`. "
        "Interpretations are based on the loadings of the current dataset and "
        "will update each time the pipeline is re-run with new data.*",
        "",
    ]

    OUT_INTERP.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ PCA interpretation saved to {OUT_INTERP}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA + clustering of tag features")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project root directory (contains data/features.csv). "
                             "Defaults to the repo data/ directory.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of families to cut the dendrogram into (default: 5).")
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of representative examples shown per family (default: 3).")
    parser.add_argument("--colors", type=str, default=None,
                        help="Comma-separated hex colors, one per family. "
                             "E.g. --colors '#FF3EA5,#00B8D9,#E6B800'. "
                             "Cycles if fewer than --k supplied.")
    parser.add_argument("--grid", type=int, default=None,
                        help="Number of columns in the sparse PCA grid. "
                             "Auto-detected from dataset size if not set.")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    project_root = args.project.expanduser().resolve() if args.project else repo_root
    _init_paths(project_root)

    k      = args.k
    n_reps = args.reps
    color_list = [c.strip().strip("'\"") for c in args.colors.split(",")] \
                 if args.colors else None
    palette = make_palette(k, color_list)
    names   = make_names(k)

    print(f"\nProject: {project_root}")
    print(f"  k={k} families  reps={n_reps}  colors={list(palette.values())}")

    if not CSV_IN.exists():
        print(f"No features CSV found at {CSV_IN}")
        print("Run: python3 pipeline/analyze.py --project <project>")
        return

    df = pd.read_csv(CSV_IN)
    print(f"\nLoaded {len(df)} tags from {CSV_IN}")

    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    missing   = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        print(f"  Warning: missing features (will skip): {missing}")

    X = df[available].fillna(0).values

    if len(df) < 2:
        print("Need at least 2 tags to cluster.")
        return

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA ──────────────────────────────────────────────────────────────────
    n_components = min(len(df), len(available), 10)
    pca      = PCA(n_components=n_components)
    X_pca    = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    if len(explained) > 2:
        print(f"\n  PCA: PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%  "
              f"PC3={explained[2]:.1f}%")
    print(f"  Total variance in first 2 PCs: {sum(explained[:2]):.1f}%\n")

    # ── Clustering (computed once, shared across all plot functions) ──────────
    Z             = linkage(X_scaled, method="ward", metric="euclidean")
    family_labels = fcluster(Z, k, criterion="maxclust")

    # ── Feature descriptions ──────────────────────────────────────────────────
    FEATURE_DESC = {
        "aspect_ratio":          "tag width-to-height ratio (wide vs tall)",
        "fill_density":          "how densely the tag fills its bounding box",
        "compactness":           "overall circularity of the form",
        "solidity":              "solidity of the largest component vs its convex hull",
        "euler_number":          "number of topological holes (O, A, 4 have holes; I, L don't)",
        "eccentricity":          "how elongated vs circular the main form is",
        "n_components":          "number of disconnected strokes or parts",
        "stroke_width_cv":       "consistency of stroke width (low = even, high = variable)",
        "stroke_width_norm":     "typical stroke thickness relative to tag size",
        "skeleton_density":      "total path length relative to tag size",
        "skeleton_to_area":      "how thin vs fat the strokes are overall",
        "skeleton_to_perimeter": "path efficiency — internal skeleton vs outer edge",
        "branching_density":     "how frequently strokes split or intersect",
        "n_loops_est":           "estimated number of enclosed areas in the stroke skeleton",
        "endpoint_branch_ratio": "flowing/cursive structure vs complex intersecting structure",
        "svg_closed_ratio":      "proportion of closed (loop) shapes in the vector trace",
        "perimeter_norm":        "total outline length relative to tag size (jaggedness proxy)",
    }

    # ── Loadings frame ────────────────────────────────────────────────────────
    n_pcs    = min(3, n_components)
    loadings = pd.DataFrame(
        pca.components_[:n_pcs].T,
        index=available,
        columns=[f"PC{i+1}" for i in range(n_pcs)]
    )
    print("  Top features by |PC1| loading:")
    top = loadings["PC1"].abs().sort_values(ascending=False).head(6)
    for feat, val in top.items():
        sign = "+" if loadings.loc[feat, "PC1"] > 0 else "-"
        print(f"    {sign}{val:.3f}  {feat}")

    # ── All plots ─────────────────────────────────────────────────────────────
    print()
    plot_pca(df, X_pca, explained)
    plot_pca_full(df, X_pca, explained)
    plot_pca_preview(df, X_pca, explained, family_labels, loadings,
                     palette=palette, names=names)
    plot_pca_sparse(df, X_pca, explained, family_labels,
                    palette=palette, names=names, grid_cols=args.grid)
    plot_dendrogram_full(df, X_scaled, Z, family_labels,
                         palette=palette)
    plot_dendrogram_preview(df, X_scaled, Z, family_labels,
                            palette=palette, names=names, n_reps=n_reps)

    # ── PCA interpretation document ───────────────────────────────────────────
    _write_pca_interpretation(loadings, explained, FEATURE_DESC)

    # ── Cluster summary ───────────────────────────────────────────────────────
    print("\n  Tags in PC1 order (low → high):")
    order = np.argsort(X_pca[:, 0])
    for i in order:
        print(f"    {X_pca[i,0]:+.2f}  {df['stem'].iloc[i]}")

    print(f"\n✓ All plots saved to {PLOT_DIR}\n")

    # ── Stage and export (repo-only — skipped for external projects) ─────────────
    import subprocess, sys
    repo_root = Path(__file__).parent.parent
    export_py = repo_root / "pipeline" / "export.py"
    export_sh = repo_root / "scripts"  / "export_for_site.sh"

    if project_root != repo_root:
        print("  (external project — skipping site export step)")
        return

    if export_py.exists():
        subprocess.run([sys.executable, str(export_py)], check=False)
    else:
        print("  (export.py not found — skipping staging)")

    if export_sh.exists():
        result = subprocess.run(["bash", str(export_sh)], capture_output=False)
        if result.returncode != 0:
            print("  (site export failed — run ./scripts/export_for_site.sh manually)")
    else:
        print("  (export_for_site.sh not found — skipping site copy)")


if __name__ == "__main__":
    main()
