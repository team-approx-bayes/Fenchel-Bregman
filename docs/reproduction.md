# 🧪 Detailed experiment reproduction

For installation and a minimal evaluation example, see the [Quick start](../README.md#-quick-start).

> 📍 Run all commands from the repository root.
All code paths below are relative to the repository root.

## 🧩 Recursive reasoning: Maze-Hard and Sudoku-Extreme

Each task directory contains launchers for the six code-evaluated methods in this release.
The HRM values in Table 1 are paper-reported, so this repository does not provide an HRM launcher.

| Method | Maze-Hard | Sudoku-Extreme |
| --- | --- | --- |
| FPRM | `rrm/sh/maze_hard/fprm.sh` | `rrm/sh/sudoku_extreme/fprm.sh` |
| GRAM | `rrm/sh/maze_hard/gram.sh` | `rrm/sh/sudoku_extreme/gram.sh` |
| TRM | `rrm/sh/maze_hard/trm.sh` | `rrm/sh/sudoku_extreme/trm.sh` |
| PTRM | `rrm/sh/maze_hard/ptrm.sh` | `rrm/sh/sudoku_extreme/ptrm.sh` |
| W-PTRM | `rrm/sh/maze_hard/w_ptrm.sh` | `rrm/sh/sudoku_extreme/w_ptrm.sh` |
| FBI (FB + IVON) | `rrm/sh/maze_hard/fb.sh` | `rrm/sh/sudoku_extreme/fb.sh` |

Use `rrm/sh/maze_hard/fb_fprm.sh` for the Maze-Hard FPRM-backbone FBI experiment.

`GRAM`, `FB`, and `FB-FPRM` train before evaluation when `CHECKPOINT` is unset.
Maze-Hard GRAM and both FB backbones read their initialization from `BASE_CHECKPOINT`.
Sudoku-Extreme GRAM trains from scratch.

```bash
DATASET=/path/to/sudoku-extreme \
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda:0 \
bash rrm/sh/sudoku_extreme/fb.sh
```

`fb_fprm.sh` uses eight training processes by default.
Set `NPROC_PER_NODE` to change that count.

The sharded launchers process shards in sequence on `DEVICE`.
They preserve the paper's shard boundaries and sampling resets without managing several GPU processes in shell.
Single runs write `evaluation/metrics.json`.
Sharded runs write `evaluation/aggregate/metrics.json`.
Each run records the measured selected accuracy and Pass@K in its output directory.

## 🎨 ARC-AGI-1 and ARC-AGI-2

The PTRM and W-PTRM launchers for ARC-AGI-1 and ARC-AGI-2 use a dedicated evaluator in `rrm/arc.py` for augmentation inversion, Q-aware voting, task-normalized Pass@K, and submission generation.
`DATASET` must contain `test/dataset.json`, the five `test/all__*.npy` arrays, `identifiers.json`, and `test_puzzles.json` produced by the ARC preprocessing pipeline.

| Method | ARC-AGI-1 | ARC-AGI-2 |
| --- | --- | --- |
| PTRM | `rrm/sh/agi-1/ptrm.sh` | `rrm/sh/agi-2/ptrm.sh` |
| W-PTRM | `rrm/sh/agi-1/w_ptrm.sh` | `rrm/sh/agi-2/w_ptrm.sh` |

The launchers use eight `torchrun` processes, 25 candidates, and inference depth 16 by default.
PTRM defaults to global batch size 32 and latent-noise scale 0.2, while W-PTRM defaults to global batch size 768 and parameter-perturbation scale 0.3.
Set `NPROC_PER_NODE`, `CANDIDATE_COUNT`, `DEPTH`, `GLOBAL_BATCH_SIZE`, `SEED`, `LATENT_NOISE_SCALE`, or `PARAMETER_PERTURBATION_SCALE` to override these values.
Set `CONFIG` only when `all_config.yaml` is not next to the checkpoint, and set `GPU_IDS` to restrict the visible GPUs.

```bash
DATASET=/path/to/preprocessed-arc-agi-1 \
CHECKPOINT=/path/to/checkpoint.pt \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash rrm/sh/agi-1/ptrm.sh
```

Complete runs write `metrics.json`, `submission.json`, `resolved_config.json`, per-rank runtime files, and rank prediction files under `OUTPUT_DIR`.

## 💬 Autoregressive LM reproduction

The Qwen3-4B-Base experiment uses the shared FB finite-$K$ weighting and failure EMA in `fb.py`, the LM-specific FBI state in `auto_lm/fbi.py`, and posterior sampling from `optim/fbi_ivon.py`.
The `auto_lm/MMPO/` directory contains the MMPO/verl runtime adapted for this experiment.
The model, datasets, checkpoints, and generated outputs are external to this repository.

Use a separate CUDA environment with PyTorch, Ray, vLLM, and eight GPUs in total, either on one machine or across two four-GPU nodes.
Install the local runtime on every Ray node, with the same repository and external input paths available on each node:

```bash
python -m pip install -r auto_lm/MMPO/requirements.txt
python -m pip install -e 'auto_lm/MMPO[vllm]'
```

Follow the [autoregressive reproduction guide](../auto_lm/README.md) for the pinned data source and checksum, evaluation-suite preparation, two-stage training commands, and posterior evaluation.
Stage 2 resumes the stage-1 checkpoint at step 200 and trains through step 300.
Evaluation draws sixteen independent IVON posterior samples and writes `evaluation/metrics.json` and `evaluation/candidates.jsonl` with Pass@1, Pass@4, Pass@8, and Pass@16 aggregation.
Add `--dry-run` to any `auto_lm/run.py` command to inspect the resolved configuration without starting Ray.
