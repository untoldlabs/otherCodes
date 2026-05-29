"""
replicates.py — Run vlm.py N times and compare symbol detection consistency.

Runs the full VLM pipeline N times independently (each saved to its own
vlm_scores_run{N}.csv and vlm_run{N}/ folder), then analyses how consistently
each symbol is detected across runs for each mark.

Why: at temperature=0 results should be deterministic, but floating-point
differences, prompt tokenisation, and model state can introduce variance.
This script measures that variance so we know which detections to trust.

Usage:
    conda activate othercodes
    python3 pipeline/replicates.py --project ~/path/to/project
    python3 pipeline/replicates.py --project ~/path/to/project --runs 5 --stems IMG_0354 IMG_2003
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

CODE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_ROOT))


def run_vlm(project_root: Path, run_id: int, stems: list[str] | None,
            model: str) -> bool:
    """Invoke vlm.py as a subprocess for one replicate run."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "vlm.py"),
        "--project", str(project_root),
        "--run-id", str(run_id),
        "--model", model,
    ]
    if stems:
        cmd += ["--stems"] + stems

    print(f"\n{'='*60}")
    print(f"  RUN {run_id}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def compare_runs(project_root: Path, n_runs: int) -> None:
    """Load all run CSVs and compute per-stem per-symbol agreement."""
    data_dir = project_root / "data"

    # Gather all run CSVs
    run_rows: dict[int, list[dict]] = {}
    for i in range(1, n_runs + 1):
        path = data_dir / f"vlm_scores_run{i}.csv"
        rows = load_csv(path)
        if rows:
            run_rows[i] = rows
        else:
            print(f"  ⚠ Run {i} CSV not found or empty: {path}")

    if not run_rows:
        print("No run data found — nothing to compare.")
        return

    # Get symbol keys from column headers
    sample_row = next(iter(run_rows.values()))[0]
    sym_keys = [k.replace("sym_", "") for k in sample_row if k.startswith("sym_")]

    # Build index: stem → run_id → {sym_key: detected (0/1)}
    stem_run_detections: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    for run_id, rows in run_rows.items():
        for row in rows:
            stem = row["stem"]
            stem_run_detections[stem][run_id] = {
                k: 1 if float(row.get(f"sym_{k}", 0) or 0) > 0 else 0
                for k in sym_keys
            }

    stems = sorted(stem_run_detections.keys())
    n_valid_runs = len(run_rows)

    # ── Per-stem summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  REPLICATE COMPARISON  ({n_valid_runs} runs, {len(stems)} stems)")
    print(f"{'='*60}\n")

    comparison_rows = []

    for stem in stems:
        run_data = stem_run_detections[stem]
        available_runs = sorted(run_data.keys())

        print(f"  {stem}")

        row = {"stem": stem}
        any_variation = False

        for key in sym_keys:
            counts = [run_data[r].get(key, 0) for r in available_runs]
            total  = sum(counts)
            agree  = (total == 0 or total == len(available_runs))
            rate   = total / len(available_runs)

            row[f"sym_{key}_count"] = total
            row[f"sym_{key}_rate"]  = round(rate, 2)
            row[f"sym_{key}_stable"] = int(agree)

            if total > 0:
                marker = "✓" if agree else "~"
                print(f"    {marker} {key:<20} detected {total}/{len(available_runs)} runs"
                      + ("  ← VARIABLE" if not agree else ""))
                if not agree:
                    any_variation = True

        if not any_variation:
            print("    (all detections fully consistent across runs)")
        print()

        comparison_rows.append(row)

    # ── Save comparison CSV ────────────────────────────────────────────────────
    if comparison_rows:
        out_path = data_dir / "vlm_replicate_comparison.csv"
        cols = ["stem"] + [c for c in comparison_rows[0] if c != "stem"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(comparison_rows)
        print(f"  ✓ Comparison saved → {out_path}")

    # ── Overall stability summary ──────────────────────────────────────────────
    total_checks = 0
    stable_checks = 0
    for stem in stems:
        run_data = stem_run_detections[stem]
        available_runs = sorted(run_data.keys())
        for key in sym_keys:
            counts = [run_data[r].get(key, 0) for r in available_runs]
            total  = sum(counts)
            if total > 0:
                total_checks  += 1
                stable_checks += int(total == len(available_runs))

    if total_checks:
        pct = 100 * stable_checks / total_checks
        print(f"  Overall stability: {stable_checks}/{total_checks} detections "
              f"consistent across all runs ({pct:.0f}%)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run vlm.py N times and compare detection consistency")
    parser.add_argument("--project", type=Path, required=True,
                        help="Project root (same as vlm.py --project)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of replicate runs (default: 5)")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Process only these stems (default: all)")
    parser.add_argument("--model", default="qwen2.5vl:7b",
                        help="Ollama model name")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip running — just compare existing run CSVs")
    args = parser.parse_args()

    project_root = args.project.expanduser().resolve()

    if not args.compare_only:
        print(f"\nRunning {args.runs} replicates on {project_root}")
        if args.stems:
            print(f"Stems: {args.stems}")

        for i in range(1, args.runs + 1):
            ok = run_vlm(project_root, i, args.stems, args.model)
            if not ok:
                print(f"\n✗ Run {i} failed — stopping.")
                sys.exit(1)

    compare_runs(project_root, args.runs)


if __name__ == "__main__":
    main()
