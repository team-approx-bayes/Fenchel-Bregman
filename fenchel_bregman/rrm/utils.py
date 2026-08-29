"""Shared contracts and runtime helpers for recursive reasoning models."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Protocol

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class TrainingOutput:
    """Model-normalized values consumed by the common training loop."""

    state: object
    per_example_task_loss: Tensor
    auxiliary_loss: Tensor
    predictions: Tensor
    selection_score: Tensor
    metrics: dict[str, Tensor]


@dataclass
class EvaluationOutput:
    """Model-normalized values consumed by the common evaluator."""

    predictions: Tensor
    selection_score: Tensor
    halted: Tensor
    steps: Tensor


@dataclass(frozen=True)
class PuzzleSplit:
    """Memory-mapped public puzzle split in stable source order."""

    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    source_ids: np.ndarray
    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(device: str) -> DistributedContext:
    """Initialize a torchrun process group or a one-process context."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return DistributedContext(0, 1, 0, torch.device(device))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank environment")
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Multi-process paper training requires CUDA")
    resolved = torch.device("cuda", local_rank)
    torch.cuda.set_device(resolved)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return DistributedContext(rank, world_size, local_rank, resolved)


def sum_distributed_gradients(
    model: nn.Module, context: DistributedContext
) -> None:
    if context.world_size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def gather_rank_ordered_vectors(
    *values: Tensor, context: DistributedContext
) -> tuple[Tensor, ...]:
    if context.world_size == 1:
        return tuple(values)
    outputs: list[Tensor] = []
    for value in values:
        if value.ndim != 1:
            raise ValueError("distributed tracker values must be vectors")
        shards = [torch.empty_like(value) for _ in range(context.world_size)]
        dist.all_gather(shards, value.contiguous())
        outputs.append(torch.cat(shards, dim=0))
    return tuple(outputs)


class ModelAdapter(Protocol):
    """Interface implemented by every model family."""

    name: str

    def build_model(
        self, task: str, preset: str, metadata: Mapping[str, Any]
    ) -> nn.Module: ...

    def load_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> Mapping[str, Any]: ...

    def training_forward(
        self,
        model: nn.Module,
        state: object,
        batch: Mapping[str, Tensor],
    ) -> TrainingOutput: ...

    def evaluation_forward(
        self, model: nn.Module, batch: Mapping[str, Tensor]
    ) -> EvaluationOutput: ...

    def build_optimizers(
        self, model: nn.Module, preset: Mapping[str, Any]
    ) -> tuple[Optimizer, Optimizer | None]: ...


_MODEL_NAMES = frozenset(("fprm", "ptrm", "gram"))


def get_model_adapter(name: str) -> ModelAdapter:
    """Load a model adapter without importing every model eagerly."""

    if name not in _MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(_MODEL_NAMES)}")
    module = importlib.import_module(f"rrm.{name}")
    try:
        return module.ADAPTER
    except AttributeError as error:
        raise RuntimeError(f"rrm.{name} does not expose ADAPTER") from error


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a regular file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically write a torch checkpoint in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_all(seed: int) -> None:
    """Seed the Python, NumPy, CPU torch, and visible CUDA streams."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream used by the public runners."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch", "cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def normalize_state_dict_keys(
    state: Mapping[str, Tensor], *, prefixes: tuple[str, ...]
) -> OrderedDict[str, Tensor]:
    """Strip at most one documented wrapper prefix and reject collisions."""

    if any(not prefix for prefix in prefixes):
        raise ValueError("State-dict prefixes must be non-empty")
    ordered_prefixes = sorted(set(prefixes), key=len, reverse=True)
    normalized: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in state.items():
        normalized_key = key
        for prefix in ordered_prefixes:
            if key.startswith(prefix):
                normalized_key = key[len(prefix) :]
                break
        if normalized_key in normalized:
            raise ValueError(f"State-dict key collision after normalization: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def validate_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    model: str,
    task: str,
    fb: bool,
    preset: str,
) -> None:
    """Require an exact match between a common checkpoint and CLI intent."""

    expected = {
        "format": "RRM_CHECKPOINT_V1",
        "model": model,
        "task": task,
        "fb": fb,
        "preset": preset,
    }
    for key, expected_value in expected.items():
        if key not in payload:
            raise ValueError(f"Checkpoint metadata is missing {key!r}")
        actual_value = payload[key]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(
                f"Checkpoint metadata {key} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


_TASK_SEQUENCE_LENGTHS = {"maze-hard": 900, "sudoku-extreme": 81}


class PuzzleTrainStream:
    """Exact group-balanced stream used by the registered RRM trainers."""

    def __init__(
        self,
        dataset: Path,
        *,
        batch_size: int,
        seed: int,
        epochs_per_iteration: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if epochs_per_iteration < 1:
            raise ValueError("epochs_per_iteration must be positive")
        self.root = dataset.expanduser().resolve(strict=True)
        split = self.root / "train"
        metadata_path = split / "dataset.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Training metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadata = metadata
        sets = metadata.get("sets")
        if not isinstance(sets, list) or len(sets) != 1 or not isinstance(sets[0], str):
            raise ValueError("Public training expects exactly one named dataset set")
        prefix = split / sets[0]
        self.inputs = np.load(str(prefix) + "__inputs.npy", mmap_mode="r")
        self.labels = np.load(str(prefix) + "__labels.npy", mmap_mode="r")
        self.puzzle_identifiers = np.load(
            str(prefix) + "__puzzle_identifiers.npy"
        )
        self.puzzle_indices = np.load(str(prefix) + "__puzzle_indices.npy")
        self.group_indices = np.load(str(prefix) + "__group_indices.npy")
        if self.inputs.ndim != 2 or self.labels.shape != self.inputs.shape:
            raise ValueError("Training inputs and labels must be matching matrices")
        if self.puzzle_indices.ndim != 1 or self.group_indices.ndim != 1:
            raise ValueError("Training puzzle/group boundaries must be vectors")
        if int(self.puzzle_indices[-1]) != len(self.inputs):
            raise ValueError("Final puzzle boundary differs from training row count")
        if int(self.group_indices[-1]) != len(self.puzzle_indices) - 1:
            raise ValueError("Final group boundary differs from puzzle count")
        if len(self.puzzle_identifiers) != len(self.puzzle_indices) - 1:
            raise ValueError("puzzle_identifiers must contain one value per puzzle")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epochs_per_iteration = int(epochs_per_iteration)
        self.ignore_label_id = metadata.get("ignore_label_id")
        self.iteration = 0
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.rng = np.random.Generator(np.random.Philox(self.seed))
        self._start_iteration()

    @property
    def num_groups(self) -> int:
        return len(self.group_indices) - 1

    def _build_order(self) -> np.ndarray:
        return np.concatenate(
            [
                self.rng.permutation(self.num_groups)
                for _ in range(self.epochs_per_iteration)
            ]
        )

    def _start_iteration(self) -> None:
        self.iteration += 1
        self.rng = np.random.Generator(
            np.random.Philox(self.seed + self.iteration)
        )
        self.order = self._build_order()
        self.cursor = 0

    def next_batch(self) -> dict[str, Tensor]:
        while True:
            rows: list[np.ndarray] = []
            puzzle_ids: list[np.ndarray] = []
            group_ids: list[np.ndarray] = []
            count = 0
            while self.cursor < len(self.order) and count < self.batch_size:
                group_id = int(self.order[self.cursor])
                self.cursor += 1
                puzzle_id = int(
                    self.rng.integers(
                        int(self.group_indices[group_id]),
                        int(self.group_indices[group_id + 1]),
                    )
                )
                start = int(self.puzzle_indices[puzzle_id])
                puzzle_size = int(self.puzzle_indices[puzzle_id + 1]) - start
                take = min(puzzle_size, self.batch_size - count)
                # TinyRecursiveModels intentionally uses the global NumPy RNG
                # for augmentation selection, independently of the Philox
                # stream used for group and puzzle selection.
                selected = start + np.random.choice(
                    puzzle_size, take, replace=False
                )
                rows.append(np.asarray(selected, dtype=np.int64))
                puzzle_ids.append(np.full(take, puzzle_id, dtype=np.int64))
                group_ids.append(np.full(take, group_id, dtype=np.int64))
                count += take
            if count == self.batch_size:
                break
            # Match the registered drop-last behavior: a partial batch is
            # consumed but discarded before advancing to the next iteration.
            self._start_iteration()
        row_indices = np.concatenate(rows)
        selected_puzzles = np.concatenate(puzzle_ids)
        labels = np.asarray(self.labels[row_indices], dtype=np.int32).copy()
        if self.ignore_label_id is not None:
            labels[labels == int(self.ignore_label_id)] = -100
        return {
            "inputs": torch.from_numpy(
                np.asarray(self.inputs[row_indices], dtype=np.int32).copy()
            ),
            "labels": torch.from_numpy(labels),
            "puzzle_identifiers": torch.from_numpy(
                np.asarray(
                    self.puzzle_identifiers[selected_puzzles], dtype=np.int32
                ).copy()
            ),
            "group_ids": torch.from_numpy(np.concatenate(group_ids)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "RRM_PUZZLE_STREAM_V1",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epochs_per_iteration": self.epochs_per_iteration,
            "iteration": self.iteration,
            "cursor": self.cursor,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "numpy_random_state": copy.deepcopy(np.random.get_state()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "format": "RRM_PUZZLE_STREAM_V1",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epochs_per_iteration": self.epochs_per_iteration,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Puzzle stream {key} mismatch")
        iteration = state.get("iteration")
        cursor = state.get("cursor")
        rng_state = state.get("rng_state")
        numpy_random_state = state.get("numpy_random_state")
        if not isinstance(iteration, int) or iteration < 1:
            raise ValueError("Puzzle stream iteration must be positive")
        order_size = self.num_groups * self.epochs_per_iteration
        if not isinstance(cursor, int) or not 0 <= cursor <= order_size:
            raise ValueError("Puzzle stream cursor is invalid")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Puzzle stream RNG state is missing")
        if not isinstance(numpy_random_state, tuple):
            raise ValueError("Puzzle stream NumPy RNG state is missing")
        self.iteration = iteration
        self.rng = np.random.Generator(
            np.random.Philox(self.seed + self.iteration)
        )
        self.order = self._build_order()
        self.cursor = cursor
        self.rng.bit_generator.state = copy.deepcopy(dict(rng_state))
        np.random.set_state(copy.deepcopy(numpy_random_state))


def build_ivon_optimizer(model: nn.Module, preset: Mapping[str, Any]) -> Optimizer:
    """Build the IVON parameter groups used by the verified posterior runs."""

    try:
        from optim.ivon import IVON
    except ImportError as error:
        raise RuntimeError("IVON is required for the selected paper preset") from error
    no_decay_suffixes = tuple(
        preset.get("no_weight_decay_suffixes", ("alpha_1_param", "alpha_2_param"))
    )
    decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay: list[nn.Parameter] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        registered = getattr(parameter, "_no_weight_decay", False) or any(
            name.endswith(suffix) for suffix in no_decay_suffixes
        )
        if registered:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    groups: list[dict[str, Any]] = []
    weight_decay = float(preset.get("ivon_weight_decay", 1e-4))
    if decay:
        groups.append(
            {
                "params": decay,
                "param_names": decay_names,
                "weight_decay": weight_decay,
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "param_names": no_decay_names,
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise ValueError("IVON requires trainable dense parameters")
    return IVON(
        groups,
        lr=float(preset.get("learning_rate", preset.get("lr", 1e-4))),
        ess=float(preset.get("ivon_ess", 100_000.0)),
        hess_init=float(preset.get("ivon_hess_init", 1.0)),
        beta1=float(preset.get("beta1", 0.9)),
        beta2=float(preset.get("ivon_beta2", 0.999)),
        weight_decay=weight_decay,
        mc_samples=int(preset.get("ivon_mc_samples", 1)),
        hess_approx=str(preset.get("ivon_hess_approx", "price")),
        clip_radius=float(preset.get("ivon_clip_radius", float("inf"))),
        sync=False,
        debias=bool(preset.get("ivon_debias", True)),
        rescale_lr=bool(preset.get("ivon_rescale_lr", True)),
    )


def load_ivon_optimizer_state(
    optimizer: Optimizer,
    state: Mapping[str, Any],
    *,
    current_step: int,
) -> None:
    """Restore IVON tensors on its parameter device and preserve step count."""

    if optimizer.__class__.__name__ != "IVON":
        raise TypeError("load_ivon_optimizer_state requires IVON")
    optimizer.load_state_dict(state)
    device = getattr(optimizer, "_device")
    dtype = getattr(optimizer, "_dtype")
    for group in optimizer.param_groups:
        for field in ("hess", "momentum"):
            value = group.get(field)
            if not isinstance(value, Tensor):
                raise ValueError(f"IVON parameter group lacks {field}")
            group[field] = value.to(device=device, dtype=dtype)
    for key, value in tuple(optimizer.state.items()):
        if isinstance(value, Tensor):
            optimizer.state[key] = value.to(device=device, dtype=dtype)
    optimizer.current_step = int(current_step)  # type: ignore[attr-defined]


def load_puzzle_split(dataset: Path, *, split: str, task: str) -> PuzzleSplit:
    """Load an unaugmented evaluation split without changing row order."""

    try:
        sequence_length = _TASK_SEQUENCE_LENGTHS[task]
    except KeyError as error:
        raise ValueError(
            f"Unknown task {task!r}; expected one of {sorted(_TASK_SEQUENCE_LENGTHS)}"
        ) from error
    split_root = dataset.expanduser().resolve(strict=True) / split
    metadata_path = split_root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Puzzle split metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or metadata.get("sets") != ["all"]:
        raise ValueError("Public evaluation expects dataset.json with sets=['all']")

    def metadata_id(name: str, *, nullable: bool = False) -> int | None:
        value = metadata.get(name)
        if nullable and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Puzzle split metadata {name!r} must be an integer")
        return value

    pad_id = metadata_id("pad_id")
    ignore_label_id = metadata_id("ignore_label_id", nullable=True)
    blank_identifier_id = metadata_id("blank_identifier_id")
    assert pad_id is not None and blank_identifier_id is not None
    paths = {
        name: split_root / f"all__{name}.npy"
        for name in (
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Puzzle split is missing required arrays: {missing}")

    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    inputs = arrays["inputs"]
    labels = arrays["labels"]
    identifiers = arrays["puzzle_identifiers"]
    if inputs.ndim != 2 or inputs.shape[1] != sequence_length:
        raise ValueError(
            f"{task} inputs must have shape [N, {sequence_length}], got {inputs.shape}"
        )
    if labels.shape != inputs.shape:
        raise ValueError(f"Labels shape {labels.shape} does not match inputs {inputs.shape}")
    row_count = inputs.shape[0]
    if identifiers.shape != (row_count,):
        raise ValueError("puzzle_identifiers must contain one entry per row")

    expected_boundaries = np.arange(row_count + 1, dtype=np.int64)
    for name in ("puzzle_indices", "group_indices"):
        boundaries = np.asarray(arrays[name], dtype=np.int64)
        if not np.array_equal(boundaries, expected_boundaries):
            raise ValueError(
                f"Public evaluation requires one unaugmented row per source; {name} differs"
            )
    source_ids = np.arange(row_count, dtype=np.int64)
    return PuzzleSplit(
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=identifiers,
        source_ids=source_ids,
        pad_id=pad_id,
        ignore_label_id=ignore_label_id,
        blank_identifier_id=blank_identifier_id,
    )
