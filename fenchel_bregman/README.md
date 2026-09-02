# Posterior-decision Reasoning

## Code layout

```text
variational_reasoning/
├── optim/                     # IVON and optional experimental optimizers
└── rrm/
    ├── fprm.py                # fixed-point reasoning model
    ├── ptrm.py                # TRM and PTRM model
    ├── gram.py                # GRAM model
    ├── fenchel_bregman.py     # Fenchel-Bregman objective
    ├── train.py               # training entry point
    ├── evaluation.py          # evaluation and aggregation entry point
    ├── arc.py                 # ARC-AGI data, voting, and evaluation entry point
    ├── utils.py               # data and checkpoint utilities
    └── sh/                    # Table 1 and ARC-AGI launchers
```

The FPRM implementation follows [`nilskiKonjIzDunava/fprm`](https://github.com/nilskiKonjIzDunava/fprm). The TRM and PTRM code follows [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

## Installation

Create a CUDA-enabled PyTorch environment and install the dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Each launcher reads a preprocessed dataset directory from `DATASET`.
Evaluation checkpoints come from `CHECKPOINT`. FPRM checkpoints require the released `all_config.yaml` in the same directory as the checkpoint.
Download public FPRM files from [`fixed-point-reasoners/fprm`](https://huggingface.co/fixed-point-reasoners/fprm).

## Table 1 launchers

Each task directory contains launchers for the six code-evaluated methods in this release.
The HRM values in Table 1 are paper-reported, so this repository does not provide an HRM launcher.

| Method | Maze-Hard | Sudoku-Extreme |
| --- | --- | --- |
| FPRM | `rrm/sh/maze_hard/fprm.sh` | `rrm/sh/sudoku_extreme/fprm.sh` |
| GRAM | `rrm/sh/maze_hard/gram.sh` | `rrm/sh/sudoku_extreme/gram.sh` |
| TRM | `rrm/sh/maze_hard/trm.sh` | `rrm/sh/sudoku_extreme/trm.sh` |
| PTRM | `rrm/sh/maze_hard/ptrm.sh` | `rrm/sh/sudoku_extreme/ptrm.sh` |
| W-PTRM | `rrm/sh/maze_hard/w_ptrm.sh` | `rrm/sh/sudoku_extreme/w_ptrm.sh` |
| FB | `rrm/sh/maze_hard/fb.sh` | `rrm/sh/sudoku_extreme/fb.sh` |

Use `rrm/sh/maze_hard/fb_fprm.sh` for the Maze-Hard FPRM-backbone FB experiment.

## ARC-AGI launchers

ARC-AGI-1 and ARC-AGI-2 use a dedicated evaluator in `rrm/arc.py` for augmentation inversion, Q-aware voting, task-normalized Pass@K, and submission generation.
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

Run an evaluation with a trained checkpoint:

```bash
DATASET=/path/to/maze-hard \
CHECKPOINT=/path/to/checkpoint.pt \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda:0 \
bash rrm/sh/maze_hard/fb.sh
```

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
