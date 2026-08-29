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
    ├── utils.py               # data and checkpoint utilities
    └── sh/                    # Table 1 launchers
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