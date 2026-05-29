"""
vlm_run_all.py — Run all VLM detection scripts in sequence.

Runs each script against the same project, in order:
  1. vlm.py          — neutral open-ended (stable baseline)
  2. vlm_neutral.py  — neutral variant (separate output dir)
  3. vlm_figures.py  — figures/faces focused (certain + ambiguous)
  4. vlm_geometry.py — geometry focused (certain + ambiguous)
  5. vlm_glyphs.py   — glyphs/motifs focused (certain + ambiguous)
  6. vlm_text.py     — text/letters focused (certain + ambiguous)
  7. vlm_combined.py — channelled across all categories (certain + ambiguous)

Each script saves to its own output directory and CSV so results can be compared.
Existing outputs are preserved — use --skip-existing to skip already-processed stems.

Usage:
    conda activate othercodes
    python3 pipeline/vlm_run_all.py --project ~/path/to/project
    python3 pipeline/vlm_run_all.py --project ~/proj --stems IMG_0001 IMG_0002
    python3 pipeline/vlm_run_all.py --project ~/proj --scripts neutral figures geometry
    python3 pipeline/vlm_run_all.py --project ~/proj --skip-existing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

# All available scripts in run order
ALL_SCRIPTS = [
    ("neutral",  PIPELINE_DIR / "vlm.py"),
    ("neutral2", PIPELINE_DIR / "vlm_neutral.py"),
    ("figures",  PIPELINE_DIR / "vlm_figures.py"),
    ("geometry", PIPELINE_DIR / "vlm_geometry.py"),
    ("glyphs",   PIPELINE_DIR / "vlm_glyphs.py"),
    ("text",     PIPELINE_DIR / "vlm_text.py"),
    ("combined", PIPELINE_DIR / "vlm_combined.py"),
]

SCRIPT_NAMES = [name for name, _ in ALL_SCRIPTS]


def run_script(script_path: Path, project_root: Path,
               stems: list[str] | None, model: str,
               skip_existing: bool) -> tuple[bool, float]:
    """Run one VLM script as a subprocess. Returns (success, elapsed_seconds)."""
    cmd = [
        sys.executable, str(script_path),
        "--project", str(project_root),
        "--model", model,
    ]
    if stems:
        cmd += ["--stems"] + stems
    if skip_existing:
        cmd.append("--skip-existing")

    t0     = time.time()
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0, time.time() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Run all (or selected) VLM detection scripts in sequence")
    parser.add_argument("--project", type=Path, required=True,
                        help="Project root (contains data/rasters/ or data/segmented/)")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Process only these stems (default: all)")
    parser.add_argument("--model", default="qwen2.5vl:7b",
                        help="Ollama model name")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stems that already have a report image in each variant")
    parser.add_argument("--scripts", nargs="+", default=None,
                        choices=SCRIPT_NAMES,
                        metavar="SCRIPT",
                        help=f"Run only these scripts (choices: {', '.join(SCRIPT_NAMES)})")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve()

    # Filter to requested scripts
    scripts_to_run = ALL_SCRIPTS
    if args.scripts:
        scripts_to_run = [(name, path) for name, path in ALL_SCRIPTS
                          if name in args.scripts]

    print(f"\n{'='*60}")
    print(f"  VLM Run All")
    print(f"{'='*60}")
    print(f"  Project : {project_root}")
    print(f"  Model   : {args.model}")
    print(f"  Scripts : {[name for name, _ in scripts_to_run]}")
    if args.stems:
        print(f"  Stems   : {args.stems}")
    if args.skip_existing:
        print(f"  Mode    : skip existing")
    print()

    results = []

    for name, script_path in scripts_to_run:
        if not script_path.exists():
            print(f"\n⚠  Script not found, skipping: {script_path.name}")
            results.append((name, False, 0.0, "script not found"))
            continue

        print(f"\n{'='*60}")
        print(f"  [{name.upper()}]  {script_path.name}")
        print(f"{'='*60}\n")

        ok, elapsed = run_script(
            script_path, project_root, args.stems, args.model, args.skip_existing
        )
        status = "✓" if ok else "✗"
        results.append((name, ok, elapsed, ""))
        print(f"\n  {status} {name} finished in {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total = sum(e for _, ok, e, _ in results if ok)
    for name, ok, elapsed, note in results:
        marker = "✓" if ok else "✗"
        note_str = f"  ({note})" if note else ""
        print(f"  {marker}  {name:<12}  {elapsed:5.0f}s{note_str}")
    print(f"\n  Total wall time: {total:.0f}s")
    print()

    failed = [name for name, ok, _, _ in results if not ok]
    if failed:
        print(f"  Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
