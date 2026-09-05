"""Compute official ARC metrics offline from saved rank predictions.

The SLURM eval jobs wrote per-rank predictions to
artifacts/agi-1/{ptrm,w_ptrm}/rank_predictions/ but crashed during the final
aggregation because a few degenerate grids (from pre-fix resume progress) failed
strict validation. This script loads those predictions, sanitizes degenerate
grids (clip to [0,9] = wrong answer, not fatal), and runs the official
count-first mean-Q vote to produce metrics.json and submission.json.

Usage:
    PYTHONPATH=. python compute_final_metrics.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from rrm.arc import (
    ArcPrediction,
    aggregate_arc_predictions,
    atomic_write_json,
    load_arc_dataset,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "data" / "arc1concept-aug-1000"
METHODS = ("ptrm", "w_ptrm")
WORLD_SIZE = 4


def load_method_predictions(method: str, rank_dir: Path) -> list[ArcPrediction]:
    preds: list[ArcPrediction] = []
    total_skipped = 0
    for rank in range(WORLD_SIZE):
        candidates = [
            rank_dir / f"rank_{rank}_predictions.pkl",
            rank_dir / f"rank_{rank}_progress.pkl",
        ]
        items = None
        for path in candidates:
            if path.is_file():
                with path.open("rb") as handle:
                    data = pickle.load(handle)
                items = data if isinstance(data, list) else data["predictions"]
                break
        if items is None:
            print(f"  [warn] {method} rank {rank}: no prediction file found")
            continue
        substituted = 0
        for pr in items:
            grid = np.asarray(pr["grid"], dtype=np.int16)
            if grid.ndim != 2 or min(grid.shape) < 1:
                # Degenerate (empty) prediction: substitute a 1x1 placeholder so
                # the row stays covered. It scores as a wrong answer (a 1x1 grid
                # matches only a trivial 1x1 label), which is the correct
                # treatment for a degenerate model output.
                grid = np.zeros((1, 1), dtype=np.int16)
                substituted += 1
            grid = np.clip(grid, 0, 9).astype(np.uint8)
            preds.append(
                ArcPrediction(
                    task_name=pr["task_name"],
                    input_hash=pr["input_hash"],
                    grid=grid,
                    q_logit=float(pr["q_logit"]),
                    row_index=int(pr["row_index"]),
                )
            )
        total_skipped += substituted
    if total_skipped:
        print(f"  [info] {method}: substituted {total_skipped} degenerate grids")
    return preds


def _latest_backup_rank_dir(out_dir: Path) -> Path | None:
    """Return the backup_* dir with the most recent rank predictions, if any.

    Chooses by modification time (not name), so a timestamped full backup wins
    over an older partial one like backup_complete_ranks.
    """

    candidates = [
        p for p in out_dir.glob("backup_*") if p.is_dir() and any(p.glob("rank_*_predictions.pkl"))
    ]
    if not candidates:
        return None

    def newest_pickle_mtime(path: Path) -> float:
        return max(f.stat().st_mtime for f in path.glob("rank_*_predictions.pkl"))

    return max(candidates, key=newest_pickle_mtime)


def main() -> None:
    dataset = load_arc_dataset(DATASET)
    total = int(dataset.inputs.shape[0])
    print(f"dataset rows: {total}\n")

    for method in METHODS:
        out_dir = ROOT / "artifacts" / "agi-1" / method
        if (out_dir / "metrics.json").is_file():
            print(f"=== {method}: metrics.json already exists, skipping ===\n")
            continue
        # Read from the timestamped backup if present (safe: never touches the
        # live rank_predictions dir), else fall back to the live dir.
        rank_dir = _latest_backup_rank_dir(out_dir) or (out_dir / "rank_predictions")
        print(f"=== {method}: reading from {rank_dir} ===")
        preds = load_method_predictions(method, rank_dir)
        covered = len({p.row_index for p in preds})
        print(f"  {len(preds)} predictions, {covered}/{total} rows")
        if covered < total:
            print(f"  [skip] {total - covered} rows missing; cannot aggregate\n")
            continue
        result = aggregate_arc_predictions(preds, test_puzzles=dataset.test_puzzles)
        atomic_write_json(out_dir / "metrics.json", result.metrics)
        atomic_write_json(out_dir / "submission.json", result.submission)
        for key in sorted(result.metrics):
            print(f"  {key}: {result.metrics[key]:.4f}")
        print(f"  wrote {out_dir/'metrics.json'} and submission.json\n")


if __name__ == "__main__":
    main()
