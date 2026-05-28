#!/bin/bash
set -e
# ─────────────────────────────────────────────────────────────────────────────
# export_for_site.sh
#
# Copies website-ready outputs from outputs/site/ into the static site repo.
# This script does NOT commit or push — do that manually from the site repo.
#
# Usage:
#   ./scripts/export_for_site.sh
#
# Override site repo location:
#   SITE_REPO=/path/to/other-codes-site ./scripts/export_for_site.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGED="$ANALYSIS_REPO/outputs/site"
SITE_REPO="${SITE_REPO:-$HOME/Documents/other_codes/other-codes-site}"

# ── Checks ────────────────────────────────────────────────────────────────────

if [ ! -d "$SITE_REPO/.git" ]; then
  echo "ERROR: Site repo not found (or not a git repo): $SITE_REPO"
  echo ""
  echo "Override with: SITE_REPO=/path/to/repo ./scripts/export_for_site.sh"
  exit 1
fi

if [ ! -d "$STAGED" ]; then
  echo "ERROR: Staged outputs not found: $STAGED"
  echo "Run: python3 pipeline/export.py"
  exit 1
fi

# Check all expected files are present
MISSING=0
for FILE in pca_full.svg pca_preview.svg dendrogram_full.svg dendrogram_preview.svg metrics.csv vectors.zip README.md METHODS.md pca_interpretation.md; do
  if [ ! -f "$STAGED/$FILE" ]; then
    echo "ERROR: Missing staged file: $STAGED/$FILE"
    MISSING=1
  fi
done
[ $MISSING -eq 1 ] && exit 1

# ── Copy ──────────────────────────────────────────────────────────────────────

RESULTS="$SITE_REPO/results"
DOWNLOADS="$SITE_REPO/downloads"

mkdir -p "$RESULTS"
mkdir -p "$DOWNLOADS"

# Full plots → results/
cp "$STAGED/pca_full.svg"            "$RESULTS/pca_full.svg"
cp "$STAGED/pca_preview.svg"         "$RESULTS/pca_preview.svg"
cp "$STAGED/dendrogram_full.svg"     "$RESULTS/dendrogram_full.svg"
cp "$STAGED/dendrogram_preview.svg"  "$RESULTS/dendrogram_preview.svg"

# Datasets + docs → downloads/
cp "$STAGED/metrics.csv"             "$DOWNLOADS/metrics.csv"
cp "$STAGED/vectors.zip"             "$DOWNLOADS/vectors.zip"
cp "$STAGED/README.md"               "$DOWNLOADS/README.md"
cp "$STAGED/METHODS.md"              "$DOWNLOADS/METHODS.md"
cp "$STAGED/pca_interpretation.md"   "$DOWNLOADS/pca_interpretation.md"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "✓ Copied to site repo:"
echo "  $RESULTS/pca_full.svg"
echo "  $RESULTS/pca_preview.svg"
echo "  $RESULTS/dendrogram_full.svg"
echo "  $RESULTS/dendrogram_preview.svg"
echo "  $DOWNLOADS/metrics.csv"
echo "  $DOWNLOADS/vectors.zip"
echo "  $DOWNLOADS/README.md"
echo "  $DOWNLOADS/METHODS.md"
echo "  $DOWNLOADS/pca_interpretation.md"
echo ""
echo "Next — review and push from the site repo:"
echo ""
echo "  cd $SITE_REPO"
echo "  git status"
echo "  git add results/ downloads/"
echo '  git commit -m "Update plots and datasets"'
echo "  git push"
echo ""
