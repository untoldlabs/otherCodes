# CLAUDE.md — Instructions for Claude

This file is read by Claude at the start of every session.
For the human-readable project spec, see PROJECT.md.

---

## Project in one sentence
Graffiti tag analysis platform: photograph → segment → vectorise → measure → cluster → display at other.codes.

## How to resume
Read this file and PROJECT.md, then check `data/segmented/` to see how many tags are done.
The user will say: *"Read PROJECT.md and let's continue."*

---

## Rules Claude must follow

### Dependencies
- **Every new import must be added to `requirements.txt` immediately** — not later, not when asked.
- `setup_env.sh` must stay in sync with `requirements.txt` at all times.
- Never install something ad hoc without updating both files.

### Git
- Suggest a `git commit` after every confirmed working state.
- Remind the user to push to `untoldlabs/otherCodes` after significant sessions.
- Never commit: raw photos, model weights, segmented masks, features.csv (all in .gitignore).

### Code quality
- All scripts must be runnable with `conda activate othercodes && python3 pipeline/xxx.py` from the project root.
- Paths must use `Path(__file__).parent.parent` — never hardcoded absolute paths.
- Every script needs a docstring explaining usage at the top.
- subprocess calls must use full executable paths or handle PATH explicitly — `which vtracer` gives the full path to embed.

### Content policy
- **Never analyse, reproduce, or assist with** any mark that is a hate symbol, white supremacist or fascist iconography, slur, or targeted threat. If encountered, flag it and move to `data/excluded/`.
- **Never process photos in which a person is identifiable** (face, licence plate, etc.). If the mark is otherwise valid, ask the user to crop first.
- These rules apply to code, analysis, and any generated output — not just to raw data handling.

### Data
- Raw photos contain EXIF/GPS — never commit, never log, never display without stripping.
- **Never touch the user's original photos.** `ingest.py` copies into `data/raw/` — originals are read-only source.
- Delete = move to `data/excluded/` — never permanently delete anything.
- Multiple tags per one photo: duplicate the JPG with suffix (`_a`, `_b`) before annotating.
- GPS is extracted at ingest time into `data/metadata/{stem}.json` (gitignored — never committed).

### Communication
- Think one step ahead. If we're building X and will obviously need Y, say so now.
- If a new dependency is needed, add it to requirements.txt and setup_env.sh in the same response.
- If a decision has been made, update PROJECT.md in the same response.
- Don't wait for the user to notice gaps — proactively flag them.

---

## Environment

```bash
conda activate othercodes        # always activate first
python3 pipeline/ingest.py /path/to/photos  # copy photos into data/raw/, extract GPS→data/metadata/
python3 tools/annotate.py        # annotation tool → http://localhost:5050
python3 pipeline/analyze.py      # extract features → data/features.csv
python3 pipeline/cluster.py      # PCA + dendrogram → data/plots/
```

**Platform:** Apple Silicon M3 Max, macOS, MPS for PyTorch
**Python:** 3.11 in conda env `othercodes`
**GitHub:** https://github.com/untoldlabs/otherCodes (untoldlabs is a user account, not org)

---

## Architecture constraints

- Annotation tool: Flask on port 5050, single HTML file, no build step.
- Photo display: `<img id="photo-img">` element (CSS filters apply here only, not to overlays).
- Canvas overlays (mask, crosshairs, box) sit on transparent canvas above the img.
- `maskCanvas` is the single source of truth for the mask — read from it for save, draw from it for display.
- All pipeline scripts are standalone — no shared state, no databases yet.

## Known gotchas

- `vtracer` has no CLI binary — use the Python API: `vtracer.convert_image_to_svg_py(in_path, out_path, colormode="binary", ...)` — takes BOTH input AND output path, writes directly to file, returns nothing. Passing only one path raises "missing 1 required positional argument: 'out_path'".
- `hessian_matrix_eigvals` returns different shapes across skimage versions — use `if eigs.ndim == 3` check.
- Flask `@app.errorhandler(Exception)` must be registered, and all routes must have try/except returning JSON — never let Flask return HTML error pages to the JS client.
- `conda activate` doesn't work inside bash scripts — scripts must check `$CONDA_DEFAULT_ENV` and require the user to activate first.
- SAM2 config path: `"configs/sam2.1/sam2.1_hiera_l.yaml"` — relative to where sam2 is installed, not the project root.
- `photoImg.onload` is set in `loadPhoto()` — if reset elsewhere (e.g. resetPP), set it to null before changing src to avoid re-triggering setup.

---

## Current state (update this each session)

- **Ingest:** `pipeline/ingest.py` written — copies photos, converts HEIC, extracts GPS
- **Annotation:** 11 of 13 photos segmented (prototype batch); RF brush annotations now persisted to `data/annotations/`
- **Features:** `data/features.csv` exists with 11 tags × 30 features
- **Vectors:** SVGs not yet generated (vtracer path issue — fix with full path)
- **Clustering:** `cluster.py` written but not yet run (needs pandas install + vtracer fix)
- **Website:** not started
- **GitHub:** pushed to untoldlabs/otherCodes main

### Immediate next steps
1. Fix vtracer path in `analyze.py` (run `which vtracer` in othercodes env)
2. Install pandas (`pip install pandas`) and add to requirements ✓
3. Re-run `analyze.py` to get SVGs, then run `cluster.py`
4. Review PCA + dendrogram plots
5. Build website skeleton

---

## What this project is NOT

- Not a real-time system (batch processing is fine for now)
- Not multi-user (single researcher, local tool)
- Not a public API yet
- Not using a database yet (CSV + files is fine for prototype)
