"""Shared results table: every run appends one row to `results/runs.csv`.

Fixed column order so rows from different scripts line up; unknown keys are an
error (add the column here first) so the table can't silently drift.
"""
from __future__ import annotations

import csv
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "runs.csv"

COLUMNS = [
    "run_id", "timestamp", "git_commit", "script", "purpose",
    "model", "layer", "features", "sae_size", "sae_release", "control",
    "dataset", "subject",
    "split", "train_stories", "eval_stories", "n_voxels", "voxel_selection",
    "delays", "alphas", "seed",
    "mean_corr", "median_corr", "top1pct_corr",
    "mean_ceiling", "normalized_mean", "normalized_median",
    "p_value", "p_value_median", "n_perm", "frac_voxels_sig", "n_sig_voxels_fdr",
    "n_features_used", "layer_used", "ci_low", "ci_high", "cv",
    "elapsed_s", "notes",
]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def log_run(row: dict, path: Path = RESULTS_CSV) -> str:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}")
    row = dict(row)
    row.setdefault("run_id", uuid.uuid4().hex[:8])
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("git_commit", git_commit())
    for k, v in row.items():
        if isinstance(v, (list, tuple)):
            row[k] = " ".join(map(str, v))
        elif isinstance(v, float):
            row[k] = f"{v:.4f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_header(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row["run_id"]


def _migrate_header(path: Path) -> None:
    """Rewrite an older CSV under the current COLUMNS (new columns left blank)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        header = next(csv.reader(f), [])
    if header == COLUMNS:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
