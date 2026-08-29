# Recursive Reasoning Models

The public RRM implementation is intentionally flat:

- `fprm.py`, `ptrm.py`, and `gram.py` contain the three model families.
- `fenchel_bregman.py` contains the model-independent proposed objective.
- `train.py` and `evaluation.py` provide the shared CLIs.
- `utils.py` contains only shared runtime contracts.
- `sh/maze_hard` and `sh/sudoku_extreme` contain the Table 1 launchers.

Baseline behavior is the default. Passing `--fb` opts into the proposed
objective. See the repository [README](../README.md) for setup, checkpoint
requirements, and commands.
