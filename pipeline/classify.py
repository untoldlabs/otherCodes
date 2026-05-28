"""
classify.py — Random Forest pixel classifier for tag segmentation.

Given a photo and user-painted foreground/background strokes,
computes multi-scale image features and trains a Random Forest
to predict tag vs background for every pixel.

Each feature is tagged with a group label. After training, importances
are summed within each group and returned alongside the mask so the
caller can display which broad feature class drove the decision.

Feature groups
--------------
  colour       : raw RGB, HSV, and per-channel Gaussian blurs
  scale_space  : multi-scale blurred grayscale
  edge_texture : LoG, Sobel, Hessian eigenvalues
  sam_hint     : SAM soft-mask passed through same filter bank as gray
  roi_spatial  : EDT + 4 directional ramps inside the ROI

Inspired by ilastik / Trainable Weka Segmentation.
"""

from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import color, feature, filters
from sklearn.ensemble import RandomForestClassifier


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(img_rgb: np.ndarray,
                     sam_hint: np.ndarray = None,
                     roi_mask: np.ndarray = None):
    """
    Compute multi-scale feature stack for each pixel.

    Args:
        img_rgb:   H x W x 3 uint8 image
        sam_hint:  H x W uint8 soft mask from SAM (0–255). Passed through
                   the same multi-scale filter bank as grayscale.
        roi_mask:  H x W bool/uint8, non-zero inside the region of interest.
                   Produces 5 spatial features: central EDT + 4 directional
                   ramps (left↔right, top↔bottom), all zeroed outside the ROI.

    Returns:
        pixel_features : (H*W, n_features) float32 array
        n_features     : int
        feat_groups    : list[str] of length n_features — group label per column
    """
    img = img_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    feats  = []
    groups = []

    def add(plane, group):
        feats.append(plane.astype(np.float32))
        groups.append(group)

    # ── Colour: raw RGB + HSV ────────────────────────────────────────────────
    for c in range(3):
        add(img[:, :, c], 'colour')

    hsv = color.rgb2hsv(img)
    for c in range(3):
        add(hsv[:, :, c], 'colour')

    # ── Scale-space: multi-scale Gaussian (gray + per-channel colour) ────────
    gray = color.rgb2gray(img)
    add(gray, 'scale_space')

    for sigma in [1, 2, 4, 8]:
        add(filters.gaussian(gray, sigma=sigma), 'scale_space')
        for c in range(3):
            add(filters.gaussian(img[:, :, c], sigma=sigma), 'colour')

    # ── Edge / texture: LoG, Sobel, Hessian ──────────────────────────────────
    for sigma in [1, 2, 4]:
        add(filters.laplace(filters.gaussian(gray, sigma=sigma)), 'edge_texture')

    for sigma in [1, 2, 4]:
        add(filters.sobel(filters.gaussian(gray, sigma=sigma)), 'edge_texture')

    for sigma in [2, 4]:
        try:
            H_elems = feature.hessian_matrix(gray, sigma=sigma,
                                              use_gaussian_derivatives=True)
        except TypeError:
            H_elems = feature.hessian_matrix(gray, sigma=sigma)
        eigs = feature.hessian_matrix_eigvals(H_elems)
        if eigs.ndim == 3:
            add(eigs[0], 'edge_texture')
            add(eigs[1], 'edge_texture')
        else:
            add(eigs, 'edge_texture')

    # ── SAM hint: same multi-scale treatment as grayscale ────────────────────
    if sam_hint is not None:
        sam = sam_hint.astype(np.float32) / 255.0
        add(sam, 'sam_hint')
        for sigma in [1, 2, 4, 8]:
            add(filters.gaussian(sam, sigma=sigma), 'sam_hint')
        for sigma in [1, 2, 4]:
            add(filters.laplace(filters.gaussian(sam, sigma=sigma)), 'sam_hint')
        add(filters.sobel(sam), 'sam_hint')

    # ── ROI spatial features ─────────────────────────────────────────────────
    # 1. Central EDT  — 0 at ROI edge, ~1 at deepest interior point
    # 2–5. Directional ramps (left→right, right→left, top→bottom, bottom→top)
    # All zeroed outside the ROI mask.
    if roi_mask is not None:
        roi_binary = (roi_mask > 0).astype(np.uint8)
        roi_f      = roi_binary.astype(np.float32)

        dist     = distance_transform_edt(roi_binary)
        max_dist = dist.max()
        add((dist / max_dist) if max_dist > 0 else dist, 'roi_spatial')

        active_rows = np.where(roi_binary.any(axis=1))[0]
        active_cols = np.where(roi_binary.any(axis=0))[0]
        if active_rows.size > 0 and active_cols.size > 0:
            y_min, y_max = int(active_rows[0]),  int(active_rows[-1])
            x_min, x_max = int(active_cols[0]),  int(active_cols[-1])
            Y, X  = np.mgrid[0:h, 0:w]
            x_n   = np.clip((X - x_min) / max(x_max - x_min, 1), 0., 1.)
            y_n   = np.clip((Y - y_min) / max(y_max - y_min, 1), 0., 1.)
            add(x_n       * roi_f, 'roi_spatial')   # left → right
            add((1 - x_n) * roi_f, 'roi_spatial')   # right → left
            add(y_n       * roi_f, 'roi_spatial')   # top → bottom
            add((1 - y_n) * roi_f, 'roi_spatial')   # bottom → top
        else:
            zero = np.zeros((h, w), dtype=np.float32)
            for _ in range(4):
                add(zero, 'roi_spatial')

    stack = np.stack(feats, axis=-1)
    return stack.reshape(h * w, -1), stack.shape[2], groups


# ── Training and prediction ───────────────────────────────────────────────────

def train_and_predict(
    img_rgb: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    box: dict = None,
    sam_hint: np.ndarray = None,
    roi_mask: np.ndarray = None,
    n_estimators: int = 100,
) -> tuple:
    """
    Train a Random Forest on annotated pixels and predict the full image.

    Args:
        img_rgb:      H x W x 3 uint8 image
        fg_mask:      H x W uint8 mask, >0 where user painted foreground
        bg_mask:      H x W uint8 mask, >0 where user painted background
        box:          optional {x, y, w, h} to restrict prediction area
        sam_hint:     optional H x W uint8 SAM soft mask (used as features)
        roi_mask:     optional H x W bool/uint8 ROI polygon or bbox mask
        n_estimators: number of trees in the forest

    Returns:
        mask         : H x W uint8 prediction mask (255 = tag, 0 = background)
        importances  : dict[group_name -> fraction] summing to ~1.0,
                       sorted descending by importance
    """
    h, w = img_rgb.shape[:2]

    using_hint = sam_hint is not None
    using_roi  = roi_mask is not None
    print(f"  Computing features for {w}x{h} image"
          f"{' + SAM hint' if using_hint else ''}"
          f"{' + ROI spatial' if using_roi else ''}...", flush=True)

    pixel_features, n_feat, feat_groups = compute_features(
        img_rgb, sam_hint=sam_hint, roi_mask=roi_mask)

    extras = []
    if using_hint: extras.append("9 SAM-hint")
    if using_roi:  extras.append("5 ROI-spatial")
    extra_str = f" (inc. {', '.join(extras)})" if extras else ""
    print(f"  Feature vector: {n_feat} per pixel{extra_str}")

    # ── Build training set ───────────────────────────────────────────────────
    fg_pixels = np.where(fg_mask.ravel() > 0)[0]
    bg_pixels = np.where(bg_mask.ravel() > 0)[0]
    print(f"  Training pixels: {len(fg_pixels)} fg, {len(bg_pixels)} bg")

    if len(fg_pixels) == 0 or len(bg_pixels) == 0:
        raise ValueError("Need both foreground and background strokes to train")

    X_train = np.vstack([pixel_features[fg_pixels], pixel_features[bg_pixels]])
    y_train = np.array([1] * len(fg_pixels) + [0] * len(bg_pixels))

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"  Training RF ({n_estimators} trees)...", flush=True)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)

    # ── Feature importances grouped by category ──────────────────────────────
    raw_imp    = clf.feature_importances_          # one float per feature
    group_sums = defaultdict(float)
    for grp, imp in zip(feat_groups, raw_imp):
        group_sums[grp] += float(imp)
    total      = sum(group_sums.values()) or 1.0
    importances = dict(
        sorted({k: round(v / total, 4) for k, v in group_sums.items()}.items(),
               key=lambda x: -x[1])
    )
    imp_str = "  Importances: " + "  ".join(
        f"{k}={v:.1%}" for k, v in importances.items())
    print(imp_str)

    # ── Predict ──────────────────────────────────────────────────────────────
    if box and box.get("w", 0) > 10 and box.get("h", 0) > 10:
        cx = max(0, int(box["x"]))
        cy = max(0, int(box["y"]))
        cw = min(int(box["w"]), w - cx)
        ch = min(int(box["h"]), h - cy)
        print(f"  Predicting box region ({cw}×{ch})...", flush=True)
        rr, cc   = np.meshgrid(np.arange(cy, cy+ch), np.arange(cx, cx+cw), indexing='ij')
        box_flat = rr.ravel() * w + cc.ravel()
        preds    = clf.predict(pixel_features[box_flat])
        result   = np.zeros(h * w, dtype=np.uint8)
        result[box_flat] = (preds * 255).astype(np.uint8)
    else:
        print(f"  Predicting full image ({w*h} px)...", flush=True)
        preds  = clf.predict(pixel_features)
        result = (preds * 255).astype(np.uint8)

    print(f"  Done. Tag pixels: {(result > 0).sum()}")
    return result.reshape(h, w), importances
