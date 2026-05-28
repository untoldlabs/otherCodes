#!/bin/bash
# other.codes — install all packages into the othercodes conda environment
#
# Usage:
#   conda create -n othercodes python=3.11 -y
#   conda activate othercodes
#   bash setup_env.sh

echo ""
echo "========================================="
echo "  other.codes — installing packages"
echo "========================================="
echo ""

# Verify we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "othercodes" ]]; then
    echo "ERROR: Please activate the environment first:"
    echo "  conda activate othercodes"
    echo ""
    exit 1
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo ""

# ── Core image + science packages ─────────────────────────────────────────────
echo "Installing image + science packages..."
pip install \
    pillow \
    pillow-heif \
    opencv-python \
    numpy \
    scipy \
    scikit-image \
    scikit-learn \
    pandas \
    matplotlib \
    flask \
    vtracer

# ── PyTorch — platform-specific ────────────────────────────────────────────────
echo ""
echo "Detecting platform for PyTorch install..."

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "→ Apple Silicon Mac (MPS)"
    pip install torch torchvision

elif [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    echo "→ Intel Mac (CPU only)"
    pip install torch torchvision

elif [[ "$OS" == "Linux" ]]; then
    # Try to detect CUDA
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        echo "→ Linux with CUDA $CUDA_VER detected"
        # Pick the closest supported wheel
        if [[ "$CUDA_VER" == 12* ]]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VER" == 11* ]]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            echo "  (Unknown CUDA version — trying cu121 wheel)"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        fi
    else
        echo "→ Linux, no GPU detected — CPU only"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

else
    echo "→ Unknown platform — attempting default PyTorch install"
    pip install torch torchvision
fi

# ── SAM 2 ─────────────────────────────────────────────────────────────────────
echo ""
echo "Installing Segment Anything Model 2..."
pip install git+https://github.com/facebookresearch/sam2.git

# ── SAM 2 model weights ───────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Downloading SAM 2 weights (~900MB)"
echo "========================================="
mkdir -p models
if [ ! -f "models/sam2_hiera_large.pt" ]; then
    curl -L "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
         -o models/sam2_hiera_large.pt \
         --progress-bar
    echo "✓ Model downloaded"
else
    echo "✓ Model already present — skipping"
fi

echo ""
echo "========================================="
echo "  All done!"
echo ""
echo "  To launch the annotation tool:"
echo "    Double-click launch.command in Finder"
echo "    — or —"
echo "    python3 tools/annotate.py --project /path/to/your/project"
echo "    → http://localhost:5050"
echo ""
echo "  Feature extraction + clustering:"
echo "    → http://localhost:5050/pipeline"
echo "========================================="
echo ""
