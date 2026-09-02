# Recursive Reasoning Models

The public RRM implementation is intentionally flat:

- `fprm.py`, `ptrm.py`, and `gram.py` contain the three model families.
- `fenchel_bregman.py` contains the model-independent proposed objective.
- `train.py` and `evaluation.py` provide the shared CLIs.
- `arc.py` provides ARC-AGI dataset validation, distributed inference, voting, metrics, and submission generation.
- `utils.py` contains only shared runtime contracts.
- `sh/maze_hard` and `sh/sudoku_extreme` contain the Table 1 launchers.
- `sh/agi-1` and `sh/agi-2` contain the ARC-AGI PTRM and W-PTRM launchers.

Baseline behavior is the default.
Passing `--fb` opts into the proposed objective.
See the repository [README](../README.md) for setup, checkpoint requirements, and commands.
