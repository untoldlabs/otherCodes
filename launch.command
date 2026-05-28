#!/bin/bash
# other.codes — Launch the annotation tool
#
# Double-click this file in Finder to start.
# A terminal window opens showing the server log.
# The app opens at http://localhost:5050 in your browser.
#
# To stop: close this terminal window (Ctrl+C).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ┌─────────────────────────────┐"
echo "  │       other.codes           │"
echo "  │   graffiti tag analysis     │"
echo "  └─────────────────────────────┘"
echo ""

# ── Find conda ────────────────────────────────────────────────────────────────
CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "  ERROR: conda not found."
    echo "  Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    echo "  then run:  bash setup_env.sh"
    read -p "  Press Enter to close."
    exit 1
fi

source "$CONDA_SH"
conda activate othercodes 2>/dev/null

if [ "$CONDA_DEFAULT_ENV" != "othercodes" ]; then
    echo "  ERROR: conda environment 'othercodes' not found."
    echo "  Run this first:"
    echo "    conda create -n othercodes python=3.11 -y"
    echo "    conda activate othercodes"
    echo "    bash \"$SCRIPT_DIR/setup_env.sh\""
    read -p "  Press Enter to close."
    exit 1
fi

echo "  Environment: $CONDA_DEFAULT_ENV  ✓"
echo ""

# ── Start server ──────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
python3 tools/annotate.py &
SERVER_PID=$!

# Wait for Flask to be ready
for i in 1 2 3 4 5; do
    sleep 1
    if curl -s http://localhost:5050 > /dev/null 2>&1; then
        break
    fi
done

# Open browser
open http://localhost:5050
echo "  Opened http://localhost:5050"
echo "  Close this window (or Ctrl+C) to stop the server."
echo ""

# Keep terminal open and show server output
wait $SERVER_PID
