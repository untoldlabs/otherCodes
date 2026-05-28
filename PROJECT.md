# other.codes — Project Document
*Last updated: 2026-05-25*

---

## What This Is

A data science platform for analysing human symbolic marks — graffiti tags, cave paintings, glyphs, doodles, and any mark a human makes on a surface. Tags are the prototype dataset, but the frame is wider: **humans have been making marks since the Palaeolithic, and those marks are measurable**.

The system photographs, segments, vectorises, measures, and clusters marks to reveal style families, influences, geographic patterns, and — eventually — something about the cognitive and aesthetic universals behind mark-making itself.

The analogy: Pokémon GO, but capturing real things made by real people across all of human history — far more interesting than fake animals.

**Live domain:** [other.codes](https://other.codes)
**GitHub:** https://github.com/untoldlabs/otherCodes (private → will be public)
**Conda env:** `othercodes` (Python 3.11, Apple Silicon)

### The broader questions

This project sits at the intersection of several fields and is designed to invite collaboration:

- **Neuroscience**: is aesthetic response to marks innate and unconscious (hardwired by evolution, present across cultures) or learned and semantic (contingent on knowing what the mark means)? The pairwise beauty study is a behavioural proxy for this question. fMRI or EEG studies of mark perception would be the next step.
- **Psychology**: perception of symbols, valence, familiarity effects, the role of legibility in beauty judgements.
- **Human culture and geography**: how do mark-making styles cluster geographically? Do they diffuse like genes, like languages, or like fashion? The dendrogram and PCA are early tools for this.
- **Archaeology / cognitive evolution**: cave marks are the earliest evidence of symbolic cognition. Can the same morphological pipeline applied to contemporary graffiti say anything about the marks at Lascaux, Altamira, Blombos?
- **Semiotics**: the continuum from pure abstract mark → recognisable symbol → legible word. Where does meaning begin?

---

## Decisions Made

| Decision | Choice | Notes |
|---|---|---|
| Domain | `other.codes` | Double meaning: graffiti as coded language + programming codes |
| GitHub | `untoldlabs/otherCodes` | Under Untold Labs — cultural data science fits their scope |
| Pipeline language | Python | Best ecosystem for CV, ML, data analysis |
| Segmentation | SAM2 (Hiera Large) + RF classifier | SAM2 for objects, RF (ilastik-style) for paint strokes |
| SVG vectorisation | `vtracer` | Fast, good quality, already installed |
| Dataset (prototype) | 13 iPhone photos, 11 segmented | HEIC → JPG via pillow-heif |
| Deployment | Local pipeline → demo website | Build and validate locally first |
| Data privacy | `data/` fully gitignored | Raw photos contain EXIF/GPS — strip before any public release |
| Content policy | Exclude hate, prejudice, identifying photos | See Content Policy section below |
| Annotation tool | Flask on localhost:5050 | `python3 tools/annotate.py` |

---

## Content Policy

The dataset and any public outputs must exclude:

1. **Hate and prejudice symbols** — any mark associated with racial hatred, ethnic persecution, religious bigotry, or political violence (swastikas, white supremacist iconography, etc.). These are excluded regardless of artistic intent or historical context. Move to `data/excluded/` with reason noted.

2. **Prejudiced or targeted language** — tags that spell slurs, targeted threats, or content designed to demean a group of people.

3. **Identifying photos** — any photograph in which a person's face, licence plate, or other identifying detail is clearly visible. If the mark is otherwise interesting, crop tightly to the mark itself before ingesting, so no person is identifiable in `data/raw/`.

**Why this matters beyond ethics:** the project's scientific frame is the study of marks as abstract human symbols. Content that is primarily hate speech or identification is not a *mark* in the relevant sense — it is an act. Excluding it keeps the dataset coherent as well as responsible.

**In practice:** during annotation, if a photo contains any of the above, use the Delete button (moves to `data/excluded/`) and do not segment it. If the mark itself is fine but the surrounding photo identifies a person, duplicate and crop before annotating.

---

## Research Directions

### Data science analyses (to build)

**Semantics**
Tags exist on a spectrum from pure abstract mark to legible symbol. We want to detect and classify:
- *Linguistic*: does the tag spell a word? If so, does it carry valence (positive/negative affect, slang, provocation)? This could use OCR + sentiment models, or a manual ground-truth pass.
- *Known symbols*: circles, squares, stars, asterisks, heart symbols, arrows, hands, faces, eyes — shapes with cross-cultural legibility. Could be detected via template matching on the SVG paths, or a small classifier on shape features.
- *Pure abstraction*: tags that are signatures with no recoverable symbol or word content.
- Output: a `semantics.csv` with per-tag labels (type, detected word, valence score, symbol classes present).

**"Tag name / band name / start-up name" analysis**
The observation: these categories are more similar than people realise. "Antler", "Fevala" — words like these work simultaneously as a tag, a band name, and a startup name. They draw from the same underlying aesthetic well: brevity, strangeness, memorability, a compressed and invented identity. The cultural contexts feel very different but the quality being reached for is the same.

The research question is not "which category does this mark fit?" but rather: *what is the shared property that makes something work as a mark at all*, whether it is painted on a wall, on a poster, or on a business card? The convergence across these supposedly distinct domains is itself the finding.

This connects the project to linguistics (phonaesthetics, invented words, brand name research), design (logo and identity theory), and cultural studies (how marks travel across subcultures). It is also a good public-facing hook — people intuitively recognise the quality being described even if they have never articulated it.

**Medium / material**
The texture of the mark encodes something about what made it:
- Posca / paint pen: clean edges, even coverage, fine detail
- Spray can: soft feathered edge, aerosol gradient, overspray
- Roller or house paint: broad flat strokes, low texture
- Marker: bleeds slightly, slightly transparent on porous surfaces
- Could be predicted from texture features on the `*_isolated.png` crops (Gabor, LBP, stroke edge softness).
- Ground truth: manual annotation per tag, saved as `data/medium.csv`.
- This is its own publishable finding: can you tell a Posca from a spray can by shape alone?

**RF feature importance study**
`data/rf_importances.csv` accumulates one row per RF segmentation run with grouped feature importances (colour, scale_space, edge_texture, sam_hint, roi_spatial). With enough tags, this becomes a study in its own right: which visual properties are most distinctive about graffiti tags vs. their backgrounds? Does edge/texture dominate on brick walls but colour dominate on painted shutters? Does SAM reliability vary by surface type?

---

### Beauty / aesthetic annotation (human study)

**Concept:** Is aesthetic response to human marks universal and innate, or learned and culturally contingent? This is a neuroscience and psychology question that can be approached behaviourally at scale.

**Method — pairwise comparison (round-robin)**
- Present two marks side-by-side, ask: *which is more beautiful?*
- Bradley-Terry or Elo scoring produces a continuous beauty ranking from binary comparisons.
- Scale: with N marks, a random subset of pairs converges well (~10N comparisons for reasonable accuracy).
- Platform: simple web form on other.codes, or Mechanical Turk / Prolific for scale and demographic reach.
- **Demographic data per assessor:** age bracket, country of origin, gender (optional), familiarity with graffiti / tagging (1–5), familiarity with visual art broadly (1–5). Short — 5 fields max, all optional.

**Analysis questions:**
- Is the beauty ranking stable across assessors? (inter-rater reliability — if high, suggests shared aesthetic grammar)
- Does demographic group predict preference? (ANOVA / mixed effects models)
- Does semantic content predict beauty? (legible words vs. abstract marks vs. recognisable symbols)
- Does morphology predict beauty? (compactness, stroke width, symmetry — measurable from the mask)
- Do "beautiful" marks cluster in PCA space, or are they stylistically diverse?
- **The neuroscience question:** if beauty correlates with morphology but not with semantic familiarity, that supports an innate visual aesthetic. If it correlates with legibility and cultural familiarity, it supports a learned one. This study is a behavioural proxy; EEG/fMRI would be the next step with a neuroscience collaborator.

**Potential collaborators:** experimental aesthetics labs (e.g. Semir Zeki, Anjan Chatterjee), cultural evolution groups, neuroaesthetics researchers.

---

### Blog posts / publications

1. **The RF classifier and the ilastik legacy**
   How we adapted Trainable Weka Segmentation philosophy to a one-off interactive tool for mark segmentation. What the feature importance data reveals about how marks differ visually from their substrates — and whether that varies by surface type or medium.

2. **STAG: a fine-tuned SAM for human symbolic marks**
   SAM2 was trained on objects with clear semantic boundaries, not paint-on-surface. Fine-tuning on accumulated masks gives a domain-specific model for marks of any era — graffiti tags, cave paintings, inscriptions. Parallel to Cellpose (biology) and MedSAM (medical imaging): same adaptation strategy, different domain. Candidate name: **STAG** (Segment-Tag).

3. **Semantics of the mark: from Lascaux to the overpass**
   Humans have made marks on surfaces for at least 60,000 years. The semiotic continuum from pure abstract shape → recognisable symbol → legible word exists in cave art as much as in contemporary tagging. Using morphological clustering to ask: are there universal symbol shapes that recur across cultures and epochs? What does the distribution of marks along the abstraction–legibility axis look like?

4. **Medium as message: predicting the tool from the mark**
   Can you tell a spray can from a Posca pen by the shape of the stroke alone? Edge softness, fill density, and gradient profile may encode the tool used. This is an unusual inverse problem: inferring physical process from geometric outcome. Interesting also for archaeology — what was used to make a given cave mark?

5. **Beauty in the eye: empirical aesthetics of human symbolic marks**
   The pairwise study. Does aesthetic preference for marks vary by culture, or converge? Is it predicted by morphology, by legibility, or by something else entirely? If preferences are cross-cultural and correlate with low-level visual features (symmetry, compactness, stroke regularity), that points toward an evolved aesthetic sense for marks. If they're culturally specific, familiarity and semantic content are doing the work. The dataset — marks + morphology + semantics + beauty scores + assessor demographics — is novel and opens collaboration with neuroaesthetics, psychology, and cognitive anthropology.

---

## Decisions Still to Make

- [ ] Website primary view: **data dashboard** / **gallery** / **map-first**?
- [ ] Web framework for site: Flask (already have it) or FastAPI + React?
- [ ] Hosting: where does other.codes live? (VPS, Vercel, Railway, etc.)
- [ ] GPS metadata: extract from EXIF or add manually per tag?
- [ ] Ingest: support any image format — currently JPG, PNG, TIFF, HEIC, WEBP, BMP are recognised by `ingest.py`. Annotate tool sidebar only globs `*.jpg` so non-JPG files won't appear until we broaden the glob to include all supported extensions. Also worth considering drag-and-drop folder ingest directly from the browser rather than typing a path.
- [ ] **PNG option for ingest:** `ingest.py` currently converts HEIC → JPEG q95 (one-time lossy step). Could optionally save as PNG (lossless) at ~4–5× larger file size. Since HEICs are already lossy, the practical difference is negligible, but PNG is the cleaner choice if storage is not a concern. Change is ~6 lines in `ingest.py` + updating `.jpg` glob patterns in `annotate.py` and `analyze.py`.

---

## Pipeline Stages

### Stage 0 — Annotation Tool (`tools/annotate.py` + `tools/static/annotate.html`)
Run: `python3 tools/annotate.py` → http://localhost:5050

**Architecture:**
- Flask server on port 5050, single-page HTML app
- Photo displayed via `<img id="photo-img">` (CSS filters apply here only)
- Overlays (mask, box, crosshairs) on transparent `<canvas id="canvas">` on top
- Brush strokes on `<canvas id="brush-canvas">` above that
- All three share the same CSS transform for zoom/pan

**Segmentation methods (tab-switched):**
- **SAM tab**: click + / − points and/or draw a box prompt → SAM2 Hiera Large (`device="mps"`)
  - "Dark tag on light background" checkbox inverts before SAM
  - Enter key triggers segmentation
- **Paint (RF) tab**: brush green on tag, red on background → Random Forest classifier
  - ilastik-style multi-scale features: RGB, HSV, Gaussian blur (σ=1,2,4,8), LoG, Sobel, Hessian eigenvalues
  - Box mode restricts RF prediction to drawn region (speed + precision)
  - sklearn RandomForestClassifier (100 trees, balanced, n_jobs=-1)

**Preprocessing panel (⚙):**
- Blur (Gaussian, 0–100), Brightness (−100 to +100), Contrast (0.1–4.0)
- Threshold (binary, with "Apply threshold" for server-side exact result)
- Invert checkbox
- CSS filters applied instantly to `<img>` for live preview; server applies exact values at segment time
- Preprocessing sent with every SAM/RF call so segmentation runs on the processed image

**Edit mask section (appears after any mask exists):**
- **Brush erase**: paint to remove mask pixels (red-ringed cursor)
- **Fill erase**: BFS flood-fill from click point, removes entire connected white region
- `maskCanvas` is the editable source of truth; `saveResult()` reads from it

**Shared controls:**
- Zoom (scroll wheel or +/− buttons), pan (P key or ✥ button)
- Invert mask button (⇄)
- Save & next → (Cmd+S): saves mask PNG + RGBA isolated PNG, auto-advances to next unsegmented photo

**Keyboard shortcuts:** f/b = fg/bg mode, x = box, p = pan, 1/2 = SAM/Paint tab, Enter = run, Cmd+Z = undo, Cmd+S = save

**Delete:** moves photo + mask to `data/excluded/` (reversible, never permanent)

**Multiple tags per photo:** hover any photo in the sidebar → click the green **+** button → creates `{stem}_0001.jpg`, `_0002`, etc. in `data/raw/` (zero-padded, auto-incremented). GPS metadata is copied. Annotate each as a separate photo.

**Output:** `data/segmented/{stem}_mask.png` (grayscale) + `{stem}_isolated.png` (RGBA)

### Stage 1 — Analyse (`pipeline/analyze.py`) ← NEXT TO BUILD
Combines raster + skeleton + vector representations.

**Raster features** (from binary mask, cropped to bbox):
- `aspect_ratio`, `fill_density`, `solidity`, `compactness`
- `eccentricity`, `euler_number` (counts holes/loops — O≠A≠I)
- `n_components` (disconnected strokes)

**Stroke features** (distance transform + skeleton):
- `mean_stroke_width`, `stroke_width_std`, `stroke_width_cv`
- `skeleton_length`, `skeleton_to_area`, `skeleton_to_perimeter`

**Topology features** (skeleton graph analysis):
- `n_branch_points` (junctions), `n_endpoints`, `n_loops_est`
- `branching_density` = branch_pts / skeleton_length

**Vector features** (vtracer SVG parse):
- `n_svg_paths`, `svg_path_complexity` (node count)
- `svg_closed_paths`, `svg_closed_ratio`

**Normalisation:** all scale-sensitive metrics normalised by √area
- `skeleton_density`, `perimeter_norm`, `stroke_width_norm`

Output: `data/features.csv` (one row per tag, all metrics)
Also runs vtracer → `data/vectors/{stem}.svg`

### Stage 2 — Cluster (`pipeline/cluster.py`) ← NEXT TO BUILD
- Load `data/features.csv`
- StandardScaler → PCA
- PCA scatter: actual tag images plotted as glyphs at their coordinates
- Hierarchical clustering → dendrogram with tag thumbnails at leaves
- Output: plots to `data/plots/`

### Stage 3 — Website
- Display tags as browsable gallery
- PCA scatter (interactive, SVG glyphs)
- Dendrogram / phylogenetic tree
- (Later) map view pinned to locations
- (Later) texture/pixel analysis using full-res PNG crops

---

## Project Structure

```
other_codes/
├── PROJECT.md
├── data/
│   ├── raw/            ← original JPGs (gitignored)
│   ├── excluded/       ← rejected photos (gitignored, reversible)
│   ├── segmented/      ← {stem}_mask.png + {stem}_isolated.png (gitignored)
│   ├── vectors/        ← {stem}.svg (gitignored, generated)
│   ├── plots/          ← PCA + dendrogram outputs
│   └── features.csv    ← all tag metrics
├── pipeline/
│   ├── convert.py      ← HEIC → JPG
│   ├── classify.py     ← RF pixel classifier (used by annotation tool)
│   ├── segment.py      ← batch SAM segmentation (legacy)
│   ├── analyze.py      ← feature extraction (TO BUILD)
│   └── cluster.py      ← PCA + dendrogram (TO BUILD)
├── tools/
│   ├── annotate.py     ← Flask annotation server
│   └── static/
│       └── annotate.html
├── website/            ← TO BUILD
├── models/             ← SAM2 weights (gitignored, ~900MB)
├── setup_env.sh        ← install packages into othercodes conda env
├── setup_mac.sh        ← full Mac setup from scratch
└── requirements.txt
```

---

## How to Resume Work with Claude

When starting a new session, say:
> *"Read PROJECT.md and let's continue the other.codes project."*

---

## Environment

```bash
conda activate othercodes
python3 tools/annotate.py        # annotation tool → http://localhost:5050
python3 pipeline/analyze.py      # extract features → data/features.csv
python3 pipeline/cluster.py      # PCA + dendrogram → data/plots/
```

---

## Log

| Date | What happened |
|---|---|
| 2026-05-22 | Project scoped, stack decided, domain confirmed as other.codes |
| 2026-05-22 | Annotation tool built: SAM2 + RF classifier + preprocessing + erasers |
| 2026-05-22 | 11 of 13 photos segmented (prototype batch) |
| 2026-05-22 | Git initialised, GitHub repo created at untoldlabs/otherCodes |
| 2026-05-22 | Analysis pipeline spec written — analyze.py + cluster.py next |
| 2026-05-26 | Annotation tool: RF brush strokes persisted + restored across sessions; box restores too |
| 2026-05-26 | SAM-hint features added to RF — SAM soft mask used as spatial prior in RF feature stack |
| 2026-05-26 | Multi-tag duplicate: green + button in sidebar creates {stem}_0001.jpg etc. |
| 2026-05-26 | ingest.py written — copies photos into data/raw/, converts HEIC→JPG, extracts GPS |
| 2026-05-26 | Annotation tool: brush min 0.5px, FG/BG overlay colour pickers + visibility toggles |
