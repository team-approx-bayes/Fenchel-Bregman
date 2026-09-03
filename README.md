# Posterior-Decision Reasoning

Research workspace for recursive reasoning models on ARC-AGI and Table 1 tasks
(Sudoku-Extreme, Maze-Hard), with a Fenchel–Bregman posterior-decision training
objective.

## Repository layout

```text
posterior-decision-reasoning/
├── setup/                      # One-command environment setup (see below)
│   ├── setup.sh                # Interactive entry point
│   ├── setup.env               # Saved configuration (created on first run; git-ignored)
│   └── scripts/
│       ├── common.sh           # Shared helpers: logging, prompts, config persistence
│       ├── clone_repos.sh      # Clone/update TinyRecursiveModels + fenchel_bregman
│       ├── install_envs.sh     # Create both uv venvs and install dependencies
│       ├── build_arc_datasets.sh       # ARC-AGI-1 + ARC-AGI-2 into DATA_ROOT
│       ├── build_table1_datasets.sh    # Sudoku-Extreme + Maze-Hard into DATA_ROOT
│       ├── build_adam_atan2.sh # Build patched adam-atan2 CUDA backend
│       └── verify_setup.sh     # End-to-end sanity checks
├── data/                       # Built datasets (default location; git-ignored)
├── TinyRecursiveModels/        # Upstream TRM (SamsungSAILMontreal) — dataset builders + TRM training
└── fenchel_bregman/            # RRM implementation: FPRM / PTRM / GRAM + FB objective
    ├── rrm/
    │   ├── fprm.py             # Fixed-point reasoning model
    │   ├── ptrm.py             # TRM and PTRM models
    │   ├── gram.py             # GRAM model
    │   ├── fenchel_bregman.py  # Model-independent FB objective
    │   ├── train.py            # Shared training CLI (maze-hard, sudoku-extreme)
    │   ├── evaluation.py       # Shared evaluation CLI (Table 1 tasks)
    │   ├── arc.py              # ARC-AGI evaluation: voting, metrics, submissions
    │   └── utils.py            # Shared runtime contracts
    ├── rrm/sh/                 # Launchers: maze_hard/, sudoku_extreme/, agi-1/, agi-2/
    └── vendor/adam_atan2/      # Patched adam-atan2 source (see "GPU notes")
```

## Quick start

```bash
bash setup/setup.sh
```

The interactive script asks what you want to configure:

1. **Dataset location** — where built datasets go (default `<repo>/data/`).
   Saved to `setup/setup.env` and reused by every later step.
2. **CUDA module** — toolkit used to compile `adam-atan2` (default `cuda/13.2.2`
   on this cluster).
3. **What to install** — full setup, or individual steps: code, environments,
   datasets (ARC and/or Table 1), `adam-atan2`, verification.

Every step is idempotent and can be run standalone:

```bash
bash setup/scripts/clone_repos.sh
bash setup/scripts/install_envs.sh
bash setup/scripts/build_arc_datasets.sh /path/to/data
bash setup/scripts/build_table1_datasets.sh /path/to/data
bash setup/scripts/build_adam_atan2.sh both
bash setup/scripts/verify_setup.sh
```

## Datasets

All datasets are produced by the TinyRecursiveModels preprocessing pipeline and
share the same on-disk format (`train/` + `test/` splits with `all__*.npy`
arrays, `dataset.json`, `identifiers.json`).

| Dataset | Directory | Size | Built by |
| --- | --- | --- | --- |
| ARC-AGI-1 | `data/arc1concept-aug-1000` | ~7 GB | `build_arc_datasets.sh` |
| ARC-AGI-2 | `data/arc2concept-aug-1000` | ~9 GB | `build_arc_datasets.sh` |
| Sudoku-Extreme | `data/sudoku-extreme-1k-aug-1000` | ~1 GB | `build_table1_datasets.sh` |
| Maze-Hard | `data/maze-30x30-hard-1k` | ~1 GB | `build_table1_datasets.sh` |

Notes:

- ARC-AGI-1/2 build from the raw Kaggle JSONs already committed under
  `TinyRecursiveModels/kaggle/combined/` — no download needed.
- Sudoku/Maze download their raw data from Hugging Face on first build.
- **Do not train on both ARC-AGI-1 and ARC-AGI-2 and evaluate both**: ARC-AGI-2
  training data contains some ARC-AGI-1 evaluation puzzles.
- "augmentation not full, only N" messages during ARC builds are expected —
  some puzzles have fewer unique augmentations than the 1000 target.

## Training

### ARC-AGI (TinyRecursiveModels)

From `TinyRecursiveModels/`, pointing `data_paths` at the built datasets:

```bash
# ARC-AGI-1 (4x H100-class GPUs, ~3 days)
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[../data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True

# ARC-AGI-2: same command with data/arc2concept-aug-1000 and
# --subsets training2 evaluation2 concept semantics (see TRM README)
```

### Table 1 tasks (fenchel_bregman)

The `rrm/sh/` launchers read `DATASET` (preprocessed directory) and
`CHECKPOINT`; `GRAM`, `FB`, and `FB-FPRM` train first when `CHECKPOINT` is
unset:

```bash
# Evaluate a trained PTRM checkpoint on Maze-Hard
DATASET=/path/to/maze-30x30-hard-1k \
CHECKPOINT=/path/to/checkpoint.pt \
OUTPUT_DIR=/path/to/output \
bash rrm/sh/maze_hard/ptrm.sh

# Train + evaluate FB on Sudoku-Extreme
DATASET=/path/to/sudoku-extreme-1k-aug-1000 \
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
OUTPUT_DIR=/path/to/output \
bash rrm/sh/sudoku_extreme/fb.sh
```

### ARC-AGI evaluation (fenchel_bregman)

`rrm/sh/agi-1/` and `rrm/sh/agi-2/` evaluate PTRM / W-PTRM checkpoints with
official augmentation voting:

```bash
DATASET=/path/to/arc1concept-aug-1000 \
CHECKPOINT=/path/to/checkpoint.pt \
GPU_IDS=0,1,2,3 \
bash rrm/sh/agi-1/ptrm.sh
```

Runs write `metrics.json`, `submission.json`, and `resolved_config.json` under
`OUTPUT_DIR` (default `fenchel_bregman/artifacts/...`).

## GPU notes (GB200 / Blackwell)

This cluster's GPUs are NVIDIA GB200 (compute capability **10.0**, i.e.
`sm_100`). Two upstream packages need patching, both handled by the setup
scripts:

- **adam-atan2 0.0.3** hardcodes `-std=c++17` (torch ≥ 2.9 headers require
  C++20) and only compiles for sm_80/86/89/90. The patched source is vendored
  at `fenchel_bregman/vendor/adam_atan2/` and installed by
  `setup/scripts/build_adam_atan2.sh`. If the CUDA module is not loaded, the
  script loads `${CUDA_MODULE}` from `setup/setup.env`.
- torch itself is installed as a CUDA-13 build (`cu130`), matching the
  cluster's `cuda/13.2.2` module.

The vendored `setup.py` supports two build variants:

| Variant | Flags | Use when |
| --- | --- | --- |
| `patched` (default) | C++20 + sm_80/86/89/90/**100/120** | torch ≥ 2.9, Blackwell GPUs (GB200) |
| `legacy` | C++17 + sm_80/86/89/90 (upstream 0.0.3) | older torch builds, pre-Blackwell GPUs |

```bash
# Default (patched) build
bash setup/scripts/build_adam_atan2.sh both

# Legacy upstream build
ADAM_ATAN2_BUILD=legacy bash setup/scripts/build_adam_atan2.sh both
```

The interactive `setup/setup.sh` menu (option 6) also asks for the variant.

If you start a fresh shell before compiling anything, re-run:

```bash
module load cuda/13.2.2   # or: source setup/setup.env && module load $CUDA_MODULE
```

## Verification

```bash
bash setup/scripts/verify_setup.sh
```

Checks both venvs, CUDA visibility, `adam_atan2`/`ivon`/`evon` imports, dataset
completeness, and that the `rrm` modules import cleanly. Exits non-zero on the
first failed check.

## References

- TRM: *Less is More: Recursive Reasoning with Tiny Networks*
  ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) —
  [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- HRM: *Hierarchical Reasoning Model* ([arXiv:2506.21734](https://arxiv.org/abs/2506.21734))
- FPRM: [fixed-point-reasoners/fprm](https://huggingface.co/fixed-point-reasoners/fprm)
