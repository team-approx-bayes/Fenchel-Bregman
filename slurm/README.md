# SLURM evaluation scripts (rikyu cluster)

Evaluation jobs for PTRM and W-PTRM on ARC-AGI-1 and ARC-AGI-2, wrapping the
launchers in `fenchel_bregman/rrm/sh/`.

## Usage

All jobs require `CHECKPOINT` (path to a trained `step_N` checkpoint file with
`all_config.yaml` alongside it, or pass `CONFIG` explicitly):

```bash
# PTRM on ARC-AGI-1
sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi1_ptrm.slurm

# W-PTRM on ARC-AGI-1
sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi1_w_ptrm.slurm

# PTRM on ARC-AGI-2
sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi2_ptrm.slurm

# W-PTRM on ARC-AGI-2
sbatch --export=ALL,CHECKPOINT=/path/to/step_N slurm/eval_agi2_w_ptrm.slurm
```

## Optional overrides (via `sbatch --export=ALL,...`)

| Variable | Default (PTRM / W-PTRM) | Meaning |
|---|---|---|
| `DATASET` | `data/arc{1,2}concept-aug-1000` | Preprocessed dataset dir |
| `OUTPUT_DIR` | `fenchel_bregman/artifacts/agi-{1,2}/{ptrm,w_ptrm}` | Where metrics/submission are written |
| `CANDIDATE_COUNT` | 25 | Number of posterior candidates |
| `DEPTH` | 16 | Recursion depth |
| `GLOBAL_BATCH_SIZE` | 32 / 768 | Global batch size |
| `LATENT_NOISE_SCALE` | 0.2 (PTRM only) | Latent noise scale |
| `PARAMETER_PERTURBATION_SCALE` | 0.2 (W-PTRM only) | Weight perturbation scale |
| `SEED` | 0 | Random seed |
| `MAX_BATCHES` | unset | Limit batches (smoke test) |
| `CONFIG` | unset | Explicit `all_config.yaml` path |
| `NPROC_PER_NODE` | `$SLURM_GPUS` | DDP processes |
| `CUDA_MODULE` | `cuda/13.2.2` | CUDA toolkit module |

## Smoke test

Run a quick 2-batch check on one GPU before the full eval:

```bash
sbatch --gpus=1 --time=00:30:00 \
  --export=ALL,CHECKPOINT=/path/to/step_N,MAX_BATCHES=2 \
  slurm/eval_agi1_ptrm.slurm
```

## Notes

- Logs go to `slurm/logs/<job-name>-<job-id>.out`.
- Results: `metrics.json`, `submission.json`, `resolved_config.json` under `OUTPUT_DIR`.
- The scripts source `_common.slurm`, which sets the rikyu proxy env vars,
  single-node NCCL/Gloo settings, loads the CUDA module, activates the
  `fenchel_bregman` uv venv python, and `cd`s into `fenchel_bregman/`.
