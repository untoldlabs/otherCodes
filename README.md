# other.codes

<p align="center">
  <img src="website/cave_animals_humans_black_transparent.png" width="420" alt="Cave paintings, humans and animals, marks made by people">
</p>

A data science platform for analysing marks made by humans: symbols, signatures, and signs left in the world. Marks are photographed in the field, segmented from their backgrounds, vectorised, measured, and clustered to reveal style families, influences, and geographic patterns.

People have always made marks. This is a tool for understanding them. Capturing real things made by real people.

**Live site (coming soon):** [other.codes](https://other.codes)  
**Built by:** [Untold Labs](https://untoldlabs.org)

---

## Quick start

**Requirements:** macOS with Apple Silicon (M1/M2/M3) and [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# 1. Clone the repo
git clone https://github.com/untoldlabs/otherCodes.git
cd otherCodes

# 2. Create the conda environment
conda create -n othercodes python=3.11 -y
conda activate othercodes

# 3. Install all dependencies + SAM2 + model weights (~900 MB download)
bash setup_env.sh
```

That's it. Once setup is done, launch the app:

```bash
# Double-click launch.command in Finder, or from the terminal:
conda activate othercodes
python3 tools/annotate.py
```

Open **http://localhost:5050** in your browser.

---

## How it works

### Code vs. data

The repo contains only code. Your photos and analysis outputs live in a separate **project directory** that you create anywhere on your computer, for example `~/Documents/my_marks/`.

The annotator is pointed at a project when you launch it:

```bash
python3 tools/annotate.py --project ~/Documents/my_marks
```

Or you can select and create projects in the browser when the tool first opens. Each project has this layout:

```
my_marks/
├── data/
│   ├── raw/          <- JPG/HEIC photos (copied in via Ingest)
│   ├── segmented/    <- binary masks + RGBA isolated images per mark
│   ├── annotations/  <- saved brush strokes and SAM points
│   ├── crops/        <- tight crop of each mark in context
│   ├── rasters/      <- cropped mask/skeleton images for analysis
│   ├── vectors/      <- vtracer SVG traces
│   ├── plots/        <- PCA and dendrogram SVGs
│   ├── metadata/     <- per-photo GPS/EXIF (never committed)
│   └── features.csv  <- one row per mark, all metrics
```

### The pipeline

```
Photograph -> Ingest -> Annotate -> Analyse -> Cluster
```

**1. Ingest**: click **+ Ingest** in the annotator header and paste the path to a folder of photos. The tool copies them into your project's `data/raw/`, converts any HEIC to JPG, and strips EXIF.

**2. Annotate**: for each photo, segment the mark from the background using one of two methods:

- **SAM tab**: click foreground/background points or draw a bounding box; powered by Meta's [Segment Anything Model 2](https://github.com/facebookresearch/sam2). Best for marks with clean edges.
- **Paint tab**: brush green over the mark and red over the background; uses a Random Forest pixel classifier with multi-scale image features. Better for faint or low-contrast marks.

A preprocessing panel (blur, brightness, contrast, threshold) helps with tricky lighting. Save with **Cmd+S**.

**3. Analyse**: click **Pipeline** in the header (or go to `http://localhost:5050/pipeline`) and click **Run Feature Extraction**. This vectorises each mark with vtracer and extracts ~30 shape and stroke metrics into `data/features.csv`.

**4. Cluster**: still in the Pipeline page, set the number of style families (k), pick a colour for each family, and click **Run Clustering**. Produces PCA scatter plots and a dendrogram in `data/plots/`.

### Keyboard shortcuts (annotator)

| Key | Action |
|-----|--------|
| `f` / `b` | Foreground / background point (SAM) |
| `x` | Bounding box mode (SAM) |
| `p` | Pan mode |
| `1` / `2` | Switch to SAM / Paint tab |
| `Enter` | Run segmentation |
| `Cmd+Z` | Undo |
| `Cmd+S` | Save and advance to next photo |

---

## Running from the terminal (advanced)

If you prefer the command line over the Pipeline page:

```bash
conda activate othercodes

# Feature extraction
python3 pipeline/analyze.py --project ~/Documents/my_marks

# Clustering (5 families, custom colours)
python3 pipeline/cluster.py \
    --project ~/Documents/my_marks \
    --k 5 \
    --reps 3 \
    --colors '#6c63ff,#ff6584,#43d9ad,#facc15,#f97316'
```

---

## Repo structure

```
otherCodes/
├── tools/
│   ├── annotate.py         <- Flask server (annotator + pipeline launcher)
│   └── static/
│       ├── annotate.html   <- annotation UI
│       └── pipeline.html   <- pipeline launcher UI
├── pipeline/
│   ├── ingest.py           <- copy + convert photos into a project
│   ├── classify.py         <- Random Forest pixel classifier
│   ├── analyze.py          <- feature extraction (shape, stroke, vector)
│   └── cluster.py          <- PCA + hierarchical clustering + plots
├── models/                 <- SAM2 weights (downloaded by setup_env.sh, gitignored)
├── setup_env.sh            <- full environment setup script
├── launch.command          <- double-click to launch on Mac
└── requirements.txt        <- Python dependencies
```

---

## Privacy and data

Raw photos contain EXIF/GPS metadata and are never committed to the repo. All photo data, masks, and derived outputs live exclusively in your project directory, not inside the repo.

"Delete" in the annotation tool moves files to `data/excluded/`. Nothing is permanently deleted.

---

## Platform notes

`setup_env.sh` detects your platform and installs the right PyTorch variant automatically. The annotator also auto-detects the best available device (MPS, CUDA, or CPU) at startup.

| Platform | GPU acceleration | Notes |
|----------|-----------------|-------|
| Apple Silicon Mac (M1/M2/M3) | MPS | Primary development platform, fully tested |
| Intel Mac | CPU only | Works, but SAM segmentation is noticeably slower |
| Linux + NVIDIA GPU | CUDA | Should work, not yet tested. Open an issue if you try it |
| Linux CPU only | CPU only | Works, but slow for SAM |
| Windows | Not supported | No `launch.command`, path assumptions |

**Intel Mac / Linux: manual PyTorch install if needed**

If the auto-detection in `setup_env.sh` picks the wrong variant, install PyTorch manually before running the rest of the script:

```bash
# Intel Mac or Linux CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Linux + CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Linux + CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then run `bash setup_env.sh` as normal. It will skip the PyTorch step if it's already installed correctly.

---

## Acknowledgements

- **[Segment Anything Model 2](https://github.com/facebookresearch/sam2)** (Meta, Apache 2.0): used for interactive segmentation
- **[vtracer](https://github.com/visioncortex/vtracer)** (MIT): raster to SVG conversion
- PyTorch, scikit-learn, scikit-image, scipy, OpenCV, matplotlib: standard scientific Python stack

## License

MIT, see [LICENSE](LICENSE).

Built by [Untold Labs](https://untoldlabs.org). If you use this in your own research or project, a credit or link back is appreciated.
