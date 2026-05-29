"""
network.py — Semantic network of graffiti marks based on shared symbols.

Reads vlm_scores.csv, builds a similarity graph where marks are connected
when they share detected symbols. Outputs a self-contained HTML/D3.js
force-directed network — each node is the actual mark image, edge thickness
reflects the number of shared symbols, and clusters emerge from the layout.

Usage:
    conda activate othercodes
    python3 pipeline/network.py --project ~/path/to/project
    python3 pipeline/network.py --project ~/path/to/project --min-shared 2
    python3 pipeline/network.py --project ~/path/to/project --run-id 5
"""

import argparse
import base64
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))


# ── Load symbol data ───────────────────────────────────────────────────────────

def load_scores(project_root: Path, run_id: int | None) -> list[dict]:
    suffix  = f"_run{run_id}" if run_id is not None else ""
    candidates = [
        project_root / "data" / f"vlm_scores{suffix}.csv",
    ]
    # Also try most recent run CSV if no specific one found
    if run_id is None:
        for i in range(10, 0, -1):
            candidates.append(project_root / "data" / f"vlm_scores_run{i}.csv")
        candidates.append(project_root / "data" / "vlm_scores.csv")

    for path in candidates:
        if path.exists():
            print(f"  Loading scores from: {path.name}")
            with open(path) as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(f"No vlm_scores CSV found in {project_root / 'data'}")


def get_sym_keys(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [k.replace("sym_", "") for k in rows[0] if k.startswith("sym_")]


def detected_symbols(row: dict, sym_keys: list[str]) -> set[str]:
    return {k for k in sym_keys if float(row.get(f"sym_{k}", 0) or 0) > 0}


# ── Load mark images ───────────────────────────────────────────────────────────

def load_image_b64(stem: str, project_root: Path) -> str | None:
    candidates = [
        project_root / "data" / "rasters"   / f"{stem}_isolated_crop.png",
        project_root / "data" / "segmented" / f"{stem}_isolated.png",
        project_root / "data" / "vlm"       / f"{stem}_report.png",
    ]
    for p in candidates:
        if p.exists():
            data = p.read_bytes()
            b64  = base64.b64encode(data).decode()
            ext  = p.suffix.lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f"data:{mime};base64,{b64}"
    return None


# ── Build network ──────────────────────────────────────────────────────────────

def build_network(rows: list[dict], sym_keys: list[str],
                  min_shared: int) -> tuple[list[dict], list[dict]]:
    """
    Returns (nodes, edges).
    Nodes: {id, stem, symbols: [...], description}
    Edges: {source, target, weight (shared symbol count), shared: [...]}
    """
    # Build per-stem symbol sets
    stem_syms: dict[str, set[str]] = {}
    stem_desc: dict[str, str]      = {}
    for row in rows:
        stem = row["stem"]
        stem_syms[stem] = detected_symbols(row, sym_keys)
        stem_desc[stem] = row.get("description", "")[:120]

    stems = sorted(stem_syms.keys())

    nodes = [
        {
            "id":          stem,
            "stem":        stem,
            "symbols":     sorted(stem_syms[stem]),
            "n_symbols":   len(stem_syms[stem]),
            "description": stem_desc[stem],
        }
        for stem in stems
    ]

    edges = []
    for a, b in combinations(stems, 2):
        shared = stem_syms[a] & stem_syms[b]
        if len(shared) >= min_shared:
            edges.append({
                "source": a,
                "target": b,
                "weight": len(shared),
                "shared": sorted(shared),
            })

    return nodes, edges


# ── HTML generation ────────────────────────────────────────────────────────────

def generate_html(nodes: list[dict], edges: list[dict],
                  images: dict[str, str], project_name: str) -> str:

    nodes_json = json.dumps(nodes, indent=2)
    edges_json = json.dumps(edges, indent=2)
    images_json = json.dumps(images, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Graffiti Semantic Network — {project_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e0e0e; font-family: monospace; color: #ccc; overflow: hidden; }}

  #canvas {{ width: 100vw; height: 100vh; }}

  .link {{
    stroke: #555;
    stroke-opacity: 0.6;
  }}

  .node-img {{
    cursor: pointer;
  }}
  .node-img:hover {{ filter: brightness(1.3); }}

  .node-ring {{
    fill: none;
    stroke-width: 2.5;
    pointer-events: none;
  }}

  #tooltip {{
    position: fixed;
    background: rgba(0,0,0,0.88);
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
    pointer-events: none;
    display: none;
    max-width: 260px;
    line-height: 1.6;
    z-index: 10;
  }}
  #tooltip .stem  {{ font-size: 13px; color: #fff; font-weight: bold; margin-bottom: 4px; }}
  #tooltip .syms  {{ color: #adf; }}
  #tooltip .desc  {{ color: #999; font-size: 10px; margin-top: 4px; }}

  #legend {{
    position: fixed;
    top: 16px;
    left: 16px;
    font-size: 11px;
    color: #888;
    line-height: 1.8;
  }}
  #legend strong {{ color: #ccc; }}

  #controls {{
    position: fixed;
    bottom: 16px;
    left: 16px;
    font-size: 11px;
    color: #666;
  }}
</style>
</head>
<body>

<svg id="canvas"></svg>

<div id="tooltip">
  <div class="stem" id="tt-stem"></div>
  <div class="syms" id="tt-syms"></div>
  <div class="desc" id="tt-desc"></div>
</div>

<div id="legend">
  <strong>Graffiti Semantic Network</strong><br>
  {project_name}<br><br>
  Nodes: marks · Images: isolated crops<br>
  Edges: shared detected symbols<br>
  Thicker edge = more shared symbols<br><br>
  Scroll to zoom · Drag to pan · Drag nodes
</div>

<div id="controls">
  Clusters emerge from symbol co-occurrence
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const NODES  = {nodes_json};
const EDGES  = {edges_json};
const IMAGES = {images_json};

const NODE_R = 42;   // node image radius
const W = window.innerWidth;
const H = window.innerHeight;

// ── Colour palette per dominant symbol ────────────────────────────────────────
const SYM_COLOUR = {{
  face:       "#e07040",
  spiral:     "#9060d0",
  arch:       "#4090d0",
  circle:     "#40b0d0",
  oval:       "#40c0b0",
  cordiform:  "#e04070",
  accent_dot: "#c0c040",
  zigzag:     "#40d080",
  drip:       "#6090e0",
  eye:        "#e09030",
  arrow:      "#d04040",
  cruciform:  "#b0b0b0",
  square:     "#80a0c0",
  triangle:   "#c080a0",
  serpentiform:"#60d0a0",
  branching:  "#90c060",
  star:       "#f0d040",
  asterisk:   "#f0a030",
  wheel:      "#f06040",
}};

function nodeColour(d) {{
  if (!d.symbols || d.symbols.length === 0) return "#444";
  for (const sym of d.symbols) {{
    if (SYM_COLOUR[sym]) return SYM_COLOUR[sym];
  }}
  return "#888";
}}

// ── SVG setup ─────────────────────────────────────────────────────────────────
const svg = d3.select("#canvas")
  .attr("width", W)
  .attr("height", H);

const g = svg.append("g");  // zoom target

// Zoom
svg.call(d3.zoom()
  .scaleExtent([0.1, 6])
  .on("zoom", (e) => g.attr("transform", e.transform))
);

// Defs: clip paths for circular node images
const defs = svg.append("defs");
NODES.forEach(d => {{
  defs.append("clipPath")
    .attr("id", `clip-${{d.id.replace(/[^a-z0-9]/gi, "_")}}`)
    .append("circle")
    .attr("r", NODE_R);
}});

// ── Force simulation ───────────────────────────────────────────────────────────
const edgeScale = d3.scaleLinear()
  .domain([1, d3.max(EDGES, e => e.weight) || 1])
  .range([1.0, 5.0]);

const sim = d3.forceSimulation(NODES)
  .force("link", d3.forceLink(EDGES)
    .id(d => d.id)
    .distance(d => Math.max(80, 180 - d.weight * 18))
    .strength(d => 0.3 + d.weight * 0.05)
  )
  .force("charge", d3.forceManyBody().strength(-220))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(NODE_R + 8));

// ── Draw edges ────────────────────────────────────────────────────────────────
const link = g.append("g").attr("class", "links")
  .selectAll("line")
  .data(EDGES)
  .join("line")
  .attr("class", "link")
  .attr("stroke-width", d => edgeScale(d.weight))
  .attr("stroke-opacity", d => 0.3 + d.weight * 0.08)
  .attr("stroke", d => {{
    // colour by most shared symbol
    const top = d.shared[0];
    return SYM_COLOUR[top] ? SYM_COLOUR[top] + "88" : "#55555588";
  }});

// ── Draw nodes ────────────────────────────────────────────────────────────────
const node = g.append("g").attr("class", "nodes")
  .selectAll("g")
  .data(NODES)
  .join("g")
  .call(d3.drag()
    .on("start", (e, d) => {{
      if (!e.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end",  (e, d) => {{
      if (!e.active) sim.alphaTarget(0);
      d.fx = null; d.fy = null;
    }})
  );

// Image (or fallback circle)
node.each(function(d) {{
  const clipId = `clip-${{d.id.replace(/[^a-z0-9]/gi, "_")}}`;
  const el = d3.select(this);
  const img = IMAGES[d.id];

  if (img) {{
    el.append("image")
      .attr("class", "node-img")
      .attr("href", img)
      .attr("x", -NODE_R)
      .attr("y", -NODE_R)
      .attr("width",  NODE_R * 2)
      .attr("height", NODE_R * 2)
      .attr("clip-path", `url(#${{clipId}})`);
  }} else {{
    el.append("circle")
      .attr("r", NODE_R)
      .attr("fill", "#333");
    el.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", "8px")
      .attr("fill", "#aaa")
      .text(d.stem.slice(-6));
  }}

  // Coloured ring
  el.append("circle")
    .attr("class", "node-ring")
    .attr("r", NODE_R)
    .attr("stroke", nodeColour(d))
    .attr("stroke-opacity", 0.85);
}});

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tooltip = document.getElementById("tooltip");

node.on("mouseover", (e, d) => {{
  document.getElementById("tt-stem").textContent = d.stem;
  document.getElementById("tt-syms").textContent =
    d.symbols.length ? d.symbols.join(", ") : "(no symbols detected)";
  document.getElementById("tt-desc").textContent = d.description || "";
  tooltip.style.display = "block";
}})
.on("mousemove", (e) => {{
  tooltip.style.left = (e.clientX + 14) + "px";
  tooltip.style.top  = (e.clientY - 10) + "px";
}})
.on("mouseout", () => {{ tooltip.style.display = "none"; }});

// ── Simulation tick ───────────────────────────────────────────────────────────
sim.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic network HTML from VLM symbol scores")
    parser.add_argument("--project", type=Path, required=True,
                        help="Project root")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Use vlm_scores_run{N}.csv (default: latest found)")
    parser.add_argument("--min-shared", type=int, default=1,
                        help="Minimum shared symbols to draw an edge (default: 1)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip embedding images (faster, smaller file)")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve()
    project_name = project_root.name

    print(f"\nProject     : {project_root}")
    print(f"Min shared  : {args.min_shared} symbol(s)")

    rows     = load_scores(project_root, args.run_id)
    sym_keys = get_sym_keys(rows)
    print(f"  {len(rows)} marks, {len(sym_keys)} symbol keys")

    nodes, edges = build_network(rows, sym_keys, args.min_shared)
    print(f"  {len(nodes)} nodes, {len(edges)} edges")

    # Load mark images
    images = {}
    if not args.no_images:
        print(f"  Loading mark images...", end=" ", flush=True)
        for node in nodes:
            img = load_image_b64(node["stem"], project_root)
            if img:
                images[node["stem"]] = img
        print(f"{len(images)}/{len(nodes)} loaded")

    html     = generate_html(nodes, edges, images, project_name)
    out_path = project_root / "data" / "network.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n✓ Network → {out_path}  ({size_mb:.1f} MB)")
    print(f"  Open in browser: open '{out_path}'\n")


if __name__ == "__main__":
    main()
