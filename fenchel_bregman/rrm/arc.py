"""ARC-AGI data contracts, evaluation, and official augmentation voting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
import yaml

from rrm import ptrm
from rrm.utils import (
    DistributedContext,
    atomic_write_json,
    initialize_distributed,
    normalize_state_dict_keys,
    seed_all,
)


ARC_MAX_GRID_SIZE = 30
ARC_PUZZLE_ID_SEPARATOR = "|||"
_DIHEDRAL_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


@dataclass(frozen=True)
class ArcDatasetMetadata:
    """Metadata required to evaluate a preprocessed ARC split."""

    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    sets: tuple[str, ...]


@dataclass(frozen=True)
class ArcDataset:
    """Memory-mapped ARC test arrays plus official voting sidecars."""

    root: Path
    metadata: ArcDatasetMetadata
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    puzzle_indices: np.ndarray
    group_indices: np.ndarray
    identifier_map: tuple[str, ...]
    test_puzzles: Mapping[str, Any]


@dataclass(frozen=True)
class ArcPrediction:
    """One selected prediction restored to its original ARC coordinates."""

    task_name: str
    input_hash: str
    grid: np.ndarray
    q_logit: float
    row_index: int | None = None


@dataclass(frozen=True)
class ArcEvaluationResult:
    """Official ARC metrics and two-attempt submission payload."""

    metrics: dict[str, float]
    submission: dict[str, list[dict[str, list[list[int]]]]]


@dataclass(frozen=True)
class ArcEvaluationConfig:
    """Resolved inputs and frozen inference settings for one ARC run."""

    task: str
    checkpoint: Path
    config: Path
    dataset: Path
    output: Path
    candidate_count: int
    depth: int
    global_batch_size: int
    seed: int
    device: str
    latent_noise_scale: float | None
    parameter_perturbation_scale: float | None
    max_batches: int | None

    @property
    def method(self) -> str:
        return "ptrm" if self.latent_noise_scale is not None else "w_ptrm"


@dataclass(frozen=True)
class ArcBatch:
    """One fixed-size local rank batch and its unpadded global row IDs."""

    row_indices: np.ndarray
    inputs: Tensor
    labels: Tensor
    puzzle_identifiers: Tensor

    @property
    def effective_count(self) -> int:
        return int(self.row_indices.size)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone ARC-AGI PTRM/W-PTRM evaluation CLI."""

    parser = argparse.ArgumentParser(
        description="Evaluate PTRM or W-PTRM on a preprocessed ARC-AGI split."
    )
    parser.add_argument("--task", required=True, choices=("arc-agi-1", "arc-agi-2"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=25)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    sampling = parser.add_mutually_exclusive_group(required=True)
    sampling.add_argument("--latent-noise-scale", type=float)
    sampling.add_argument("--parameter-perturbation-scale", type=float)
    parser.add_argument("--max-batches", type=int)
    return parser


def config_from_args(
    args: argparse.Namespace, *, world_size: int | None = None
) -> ArcEvaluationConfig:
    """Validate CLI values and resolve checkpoint-relative configuration."""

    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("world size must be positive")
    rank = int(os.environ.get("RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world size)")
    if isinstance(args.candidate_count, bool) or args.candidate_count < 1:
        raise ValueError("candidate count must be positive")
    if isinstance(args.depth, bool) or args.depth < 1:
        raise ValueError("inference depth must be positive")
    if isinstance(args.global_batch_size, bool) or args.global_batch_size < 1:
        raise ValueError("global batch size must be positive")
    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max batches must be positive")
    latent_scale = args.latent_noise_scale
    parameter_scale = args.parameter_perturbation_scale
    if (latent_scale is None) == (parameter_scale is None):
        raise ValueError("exactly one ARC sampling method must be selected")
    for name, value in (
        ("latent noise scale", latent_scale),
        ("parameter perturbation scale", parameter_scale),
    ):
        if value is not None and (not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ARC checkpoint is not a file: {checkpoint}")
    dataset = args.dataset.expanduser().resolve(strict=True)
    if not dataset.is_dir():
        raise NotADirectoryError(f"ARC dataset is not a directory: {dataset}")
    config_path = (
        args.config.expanduser()
        if args.config is not None
        else checkpoint.parent / "all_config.yaml"
    ).resolve(strict=True)
    if not config_path.is_file():
        raise FileNotFoundError(f"ARC model config is not a file: {config_path}")
    output = args.output.expanduser().resolve(strict=False)
    if rank == 0 and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    return ArcEvaluationConfig(
        task=str(args.task),
        checkpoint=checkpoint,
        config=config_path,
        dataset=dataset,
        output=output,
        candidate_count=int(args.candidate_count),
        depth=int(args.depth),
        global_batch_size=int(args.global_batch_size),
        seed=int(args.seed),
        device=str(args.device),
        latent_noise_scale=(None if latent_scale is None else float(latent_scale)),
        parameter_perturbation_scale=(
            None if parameter_scale is None else float(parameter_scale)
        ),
        max_batches=(None if args.max_batches is None else int(args.max_batches)),
    )


def iter_rank_batches(
    dataset: ArcDataset,
    config: ArcEvaluationConfig,
    *,
    rank: int,
    world_size: int,
) -> Iterator[ArcBatch]:
    """Yield the official contiguous per-rank slices of each global batch."""

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("ARC rank must be in [0, world_size)")
    if config.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    local_batch_size = config.global_batch_size // world_size
    row_count = int(dataset.inputs.shape[0])
    puzzle_boundaries = np.asarray(dataset.puzzle_indices, dtype=np.int64)
    for global_start in range(0, row_count, config.global_batch_size):
        global_stop = min(global_start + config.global_batch_size, row_count)
        local_start = global_start + rank * local_batch_size
        local_stop = min(local_start + local_batch_size, global_stop)
        if local_stop < local_start:
            local_stop = local_start
        row_indices = np.arange(local_start, local_stop, dtype=np.int64)
        inputs = np.asarray(
            dataset.inputs[local_start:local_stop], dtype=np.int32
        ).copy()
        labels = np.asarray(
            dataset.labels[local_start:local_stop], dtype=np.int32
        ).copy()
        if row_indices.size:
            puzzle_rows = (
                np.searchsorted(puzzle_boundaries, row_indices, side="right") - 1
            )
            puzzle_identifiers = np.asarray(
                dataset.puzzle_identifiers[puzzle_rows], dtype=np.int32
            ).copy()
        else:
            puzzle_identifiers = np.empty((0,), dtype=np.int32)
        if dataset.metadata.ignore_label_id is not None:
            labels[labels == dataset.metadata.ignore_label_id] = -100
        padding = local_batch_size - row_indices.size
        if padding:
            inputs = np.pad(
                inputs,
                ((0, padding), (0, 0)),
                constant_values=dataset.metadata.pad_id,
            )
            labels = np.pad(
                labels,
                ((0, padding), (0, 0)),
                constant_values=-100,
            )
            puzzle_identifiers = np.pad(
                puzzle_identifiers,
                (0, padding),
                constant_values=dataset.metadata.blank_identifier_id,
            )
        yield ArcBatch(
            row_indices=row_indices,
            inputs=torch.from_numpy(inputs),
            labels=torch.from_numpy(labels),
            puzzle_identifiers=torch.from_numpy(puzzle_identifiers),
        )


def _checkpoint_state(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("ARC checkpoint must contain a state mapping")
    if payload.get("format") == "RRM_CHECKPOINT_V1":
        state = payload.get("model_state")
    elif isinstance(payload.get("model_state_dict"), Mapping):
        state = payload.get("model_state_dict")
    elif isinstance(payload.get("model"), Mapping):
        state = payload.get("model")
    else:
        state = payload
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError("ARC checkpoint does not contain a tensor model state")
    return state  # type: ignore[return-value]


def _single_state_tensor(state: Mapping[str, Tensor], suffix: str) -> Tensor:
    matches = [value for name, value in state.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one ARC checkpoint tensor ending in {suffix!r}")
    return matches[0]


def validate_checkpoint_identifier_rows(
    state: Mapping[str, Tensor], *, expected_rows: int
) -> None:
    """Reject a checkpoint whose learned puzzle IDs do not match the data."""

    rows = int(_single_state_tensor(state, "inner.puzzle_emb.weights").shape[0])
    if rows != expected_rows:
        raise ValueError(
            f"ARC checkpoint identifier rows {rows} != dataset rows {expected_rows}"
        )


def load_arc_model(
    config: ArcEvaluationConfig,
    dataset: ArcDataset,
    *,
    device: torch.device,
    local_batch_size: int | None = None,
) -> nn.Module:
    """Strict-load an official TRM checkpoint without duplicating its tensors."""

    saved_config = yaml.safe_load(config.config.read_text(encoding="utf-8"))
    if not isinstance(saved_config, Mapping) or not isinstance(
        saved_config.get("arch"), Mapping
    ):
        raise ValueError("ARC all_config.yaml must contain an arch object")
    numpy_safe_globals = (
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
        ptrm.TinyRecursiveReasoningModel_ACTV1Carry,
        ptrm.TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    with torch.serialization.safe_globals(numpy_safe_globals):
        payload = torch.load(
            config.checkpoint,
            map_location=device,
            weights_only=True,
        )
    state = _checkpoint_state(payload)
    validate_checkpoint_identifier_rows(
        state, expected_rows=dataset.metadata.num_puzzle_identifiers
    )
    token_rows = int(
        _single_state_tensor(state, "inner.embed_tokens.embedding_weight").shape[0]
    )
    if token_rows != dataset.metadata.vocab_size:
        raise ValueError(
            f"ARC checkpoint vocabulary rows {token_rows} != dataset rows "
            f"{dataset.metadata.vocab_size}"
        )
    if local_batch_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if config.global_batch_size % world_size:
            raise ValueError("global batch size must be divisible by world size")
        local_batch_size = config.global_batch_size // world_size
    arch = dict(saved_config["arch"])
    arch.update(
        batch_size=local_batch_size,
        seq_len=dataset.metadata.seq_len,
        vocab_size=dataset.metadata.vocab_size,
        num_puzzle_identifiers=dataset.metadata.num_puzzle_identifiers,
    )
    metadata = {
        "arch": arch,
        "candidate_count": config.candidate_count,
        "latent_noise_sigma": config.latent_noise_scale or 0.0,
        "inference_depth": config.depth,
    }
    with torch.device(device):
        model = ptrm.ADAPTER.build_model(config.task, "ptrm", metadata)
    normalized = normalize_state_dict_keys(
        state,
        prefixes=("_orig_mod.model.", "_orig_mod.", "model."),
    )
    model.load_state_dict(normalized, strict=True, assign=True)
    model.eval()
    return model


def _dihedral_transform(array: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return array
    if transform_id == 1:
        return np.rot90(array, k=1)
    if transform_id == 2:
        return np.rot90(array, k=2)
    if transform_id == 3:
        return np.rot90(array, k=3)
    if transform_id == 4:
        return np.fliplr(array)
    if transform_id == 5:
        return np.flipud(array)
    if transform_id == 6:
        return array.T
    if transform_id == 7:
        return np.fliplr(np.rot90(array, k=1))
    raise ValueError(f"Unknown ARC dihedral transform: {transform_id}")


def crop_grid(sequence: np.ndarray) -> np.ndarray:
    """Crop one tokenized 30-by-30 ARC sequence to its decoded color grid."""

    encoded = np.asarray(sequence)
    if encoded.shape not in ((ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,), (30, 30)):
        raise ValueError("ARC grid sequence must contain exactly 900 tokens")
    encoded = encoded.reshape(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
    maximum_area = 0
    maximum_shape = (0, 0)
    column_limit = ARC_MAX_GRID_SIZE
    for row_count in range(1, ARC_MAX_GRID_SIZE + 1):
        for column in range(1, column_limit + 1):
            token = int(encoded[row_count - 1, column - 1])
            if token < 2 or token > 11:
                column_limit = column - 1
                break
        area = row_count * column_limit
        if area > maximum_area:
            maximum_area = area
            maximum_shape = (row_count, column_limit)
    rows, columns = maximum_shape
    return (encoded[:rows, :columns].astype(np.int16) - 2).astype(np.uint8)


def inverse_augmentation(name: str) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """Return the original ARC task name and inverse augmentation function."""

    if ARC_PUZZLE_ID_SEPARATOR not in name:
        return name, lambda grid: np.asarray(grid, dtype=np.uint8)
    parts = name.split(ARC_PUZZLE_ID_SEPARATOR)
    if len(parts) < 3:
        raise ValueError(f"Malformed ARC augmented identifier: {name!r}")
    original_name = ARC_PUZZLE_ID_SEPARATOR.join(parts[:-2])
    transform_text, permutation_text = parts[-2:]
    if (
        len(transform_text) != 2
        or not transform_text.startswith("t")
        or not transform_text[1].isdigit()
    ):
        raise ValueError(f"Malformed ARC transform identifier: {name!r}")
    transform_id = int(transform_text[1])
    if transform_id not in range(8):
        raise ValueError(f"ARC transform ID must be in [0, 7]: {name!r}")
    if (
        len(permutation_text) != 10
        or not permutation_text.isdigit()
        or set(permutation_text) != set("0123456789")
    ):
        raise ValueError(f"Malformed ARC color permutation: {name!r}")
    inverse_permutation = np.argsort(
        np.asarray([int(value) for value in permutation_text], dtype=np.uint8)
    ).astype(np.uint8)
    inverse_transform_id = _DIHEDRAL_INVERSE[transform_id]

    def restore(grid: np.ndarray) -> np.ndarray:
        values = np.asarray(grid)
        if values.ndim != 2 or np.any(values < 0) or np.any(values > 9):
            raise ValueError("ARC grids must be two-dimensional colors in [0, 9]")
        untransformed = _dihedral_transform(values, inverse_transform_id)
        return inverse_permutation[untransformed].astype(np.uint8, copy=False)

    return original_name, restore


def grid_hash(grid: np.ndarray) -> str:
    """Hash an ARC grid with its two-dimensional shape."""

    values = np.asarray(grid)
    if (
        values.ndim != 2
        or values.shape[0] > 255
        or values.shape[1] > 255
        or np.any(values < 0)
        or np.any(values > 9)
    ):
        raise ValueError("ARC grids must be two-dimensional colors in [0, 9]")
    values = values.astype(np.uint8, copy=False)
    payload = bytes(values.shape) + values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _arc_grid(payload: Any) -> np.ndarray:
    values = np.asarray(payload)
    if (
        values.ndim != 2
        or values.shape[0] > ARC_MAX_GRID_SIZE
        or values.shape[1] > ARC_MAX_GRID_SIZE
        or values.shape[0] < 1
        or values.shape[1] < 1
        or np.any(values < 0)
        or np.any(values > 9)
    ):
        raise ValueError("Official ARC grids must be non-empty colors in [0, 9]")
    return values.astype(np.uint8)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def aggregate_arc_predictions(
    predictions: Sequence[ArcPrediction],
    *,
    test_puzzles: Mapping[str, Any],
    pass_ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
    submission_k: int = 2,
) -> ArcEvaluationResult:
    """Apply the official count-first, mean-Q ARC augmentation vote."""

    if not pass_ks or any(isinstance(value, bool) or value < 1 for value in pass_ks):
        raise ValueError("ARC Pass@K values must be positive integers")
    if submission_k < 1:
        raise ValueError("ARC submission candidate count must be positive")
    if not isinstance(test_puzzles, Mapping) or not test_puzzles:
        raise ValueError("ARC test puzzles must be a non-empty mapping")

    observations: dict[
        str,
        dict[str, dict[str, list[Any]]],
    ] = {}
    for prediction in predictions:
        if prediction.task_name not in test_puzzles:
            raise ValueError(
                f"Prediction references unknown ARC task {prediction.task_name!r}"
            )
        if not isinstance(prediction.input_hash, str) or not prediction.input_hash:
            raise ValueError("ARC prediction input hash must be non-empty")
        q_logit = float(prediction.q_logit)
        if not math.isfinite(q_logit):
            raise ValueError("ARC prediction Q logit must be finite")
        grid = _arc_grid(prediction.grid)
        prediction_hash = grid_hash(grid)
        task = observations.setdefault(prediction.task_name, {})
        input_predictions = task.setdefault(prediction.input_hash, {})
        stats = input_predictions.setdefault(
            prediction_hash,
            [0, 0.0, grid.copy()],
        )
        stats[0] += 1
        stats[1] += _sigmoid(q_logit)

    correct_totals = [0.0 for _ in pass_ks]
    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    for task_name, puzzle in test_puzzles.items():
        if not isinstance(task_name, str) or not isinstance(puzzle, Mapping):
            raise ValueError("ARC test puzzle entries must map string names to objects")
        test_pairs = puzzle.get("test")
        if not isinstance(test_pairs, list) or not test_pairs:
            raise ValueError(f"ARC task {task_name!r} has no test pairs")
        task_correct = [0 for _ in pass_ks]
        task_submission: list[dict[str, list[list[int]]]] = []
        for pair in test_pairs:
            if (
                not isinstance(pair, Mapping)
                or "input" not in pair
                or "output" not in pair
            ):
                raise ValueError(f"ARC task {task_name!r} has a malformed test pair")
            input_hash = grid_hash(_arc_grid(pair["input"]))
            label_hash = grid_hash(_arc_grid(pair["output"]))
            candidates = observations.get(task_name, {}).get(input_hash)
            if not candidates:
                raise ValueError(
                    f"ARC task {task_name!r} input {input_hash} has no predictions"
                )
            ranked = sorted(
                candidates.items(),
                key=lambda item: (
                    int(item[1][0]),
                    float(item[1][1]) / int(item[1][0]),
                ),
                reverse=True,
            )
            for index, pass_k in enumerate(pass_ks):
                task_correct[index] += any(
                    prediction_hash == label_hash
                    for prediction_hash, _stats in ranked[:pass_k]
                )
            selected_grids = [stats[2] for _hash, stats in ranked[:submission_k]]
            while len(selected_grids) < submission_k:
                selected_grids.append(selected_grids[0])
            task_submission.append(
                {
                    f"attempt_{index + 1}": grid.tolist()
                    for index, grid in enumerate(selected_grids)
                }
            )
        submission[task_name] = task_submission
        for index, count in enumerate(task_correct):
            correct_totals[index] += count / len(test_pairs)

    task_count = len(test_puzzles)
    metrics = {
        f"ARC/pass@{pass_k}": correct_totals[index] / task_count
        for index, pass_k in enumerate(pass_ks)
    }
    return ArcEvaluationResult(metrics=metrics, submission=submission)


def _metadata_integer(
    payload: Mapping[str, Any], name: str, *, nullable: bool = False
) -> int | None:
    value = payload.get(name)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ARC metadata {name!r} must be an integer")
    return value


def _load_metadata(path: Path) -> tuple[ArcDatasetMetadata, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ARC dataset metadata must be a JSON object")
    sets = payload.get("sets")
    if sets != ["all"]:
        raise ValueError("ARC evaluation requires dataset sets=['all']")
    pad_id = _metadata_integer(payload, "pad_id")
    ignore_label_id = _metadata_integer(payload, "ignore_label_id", nullable=True)
    blank_identifier_id = _metadata_integer(payload, "blank_identifier_id")
    vocab_size = _metadata_integer(payload, "vocab_size")
    seq_len = _metadata_integer(payload, "seq_len")
    num_puzzle_identifiers = _metadata_integer(payload, "num_puzzle_identifiers")
    assert pad_id is not None
    assert blank_identifier_id is not None
    assert vocab_size is not None
    assert seq_len is not None
    assert num_puzzle_identifiers is not None
    if seq_len != 900 or vocab_size != 12 or num_puzzle_identifiers < 1:
        raise ValueError(
            "ARC metadata requires seq_len=900, vocab_size=12, and positive identifiers"
        )
    if (pad_id, ignore_label_id, blank_identifier_id) != (0, 0, 0):
        raise ValueError("ARC metadata special token IDs must all be zero")
    return (
        ArcDatasetMetadata(
            pad_id=pad_id,
            ignore_label_id=ignore_label_id,
            blank_identifier_id=blank_identifier_id,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            sets=("all",),
        ),
        payload,
    )


def _validate_boundaries(name: str, values: np.ndarray, *, expected_stop: int) -> None:
    boundaries = np.asarray(values)
    if (
        boundaries.ndim != 1
        or boundaries.size < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or int(boundaries[0]) != 0
        or int(boundaries[-1]) != expected_stop
        or np.any(np.diff(boundaries.astype(np.int64, copy=False)) <= 0)
    ):
        raise ValueError(
            f"ARC {name} must be strictly increasing boundaries from 0 to {expected_stop}"
        )


def _validate_token_array(name: str, values: np.ndarray, *, vocab_size: int) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"ARC {name} tokens must use an integer dtype")
    chunk_rows = 4096
    for start in range(0, int(values.shape[0]), chunk_rows):
        chunk = np.asarray(values[start : start + chunk_rows])
        if int(chunk.min()) < 0 or int(chunk.max()) >= vocab_size:
            raise ValueError(f"ARC {name} tokens must be in [0, {vocab_size - 1}]")


def _validate_test_puzzles(test_puzzles: Mapping[str, Any]) -> None:
    for task_name, puzzle in test_puzzles.items():
        if not isinstance(task_name, str) or not isinstance(puzzle, Mapping):
            raise ValueError("ARC test pair sidecar must map task names to objects")
        test_pairs = puzzle.get("test")
        if not isinstance(test_pairs, list) or not test_pairs:
            raise ValueError(f"ARC task {task_name!r} must contain a test pair")
        for pair in test_pairs:
            if (
                not isinstance(pair, Mapping)
                or "input" not in pair
                or "output" not in pair
            ):
                raise ValueError(f"ARC task {task_name!r} has a malformed test pair")
            try:
                _arc_grid(pair["input"])
                _arc_grid(pair["output"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"ARC task {task_name!r} has a malformed test pair"
                ) from error


def load_arc_dataset(path: Path) -> ArcDataset:
    """Load and validate one official-format preprocessed ARC test split."""

    root = path.expanduser().resolve(strict=True)
    test_root = root / "test"
    metadata_path = test_root / "dataset.json"
    identifiers_path = root / "identifiers.json"
    test_puzzles_path = root / "test_puzzles.json"
    required = (metadata_path, identifiers_path, test_puzzles_path)
    for required_path in required:
        if not required_path.is_file():
            raise FileNotFoundError(f"ARC dataset file is missing: {required_path}")

    metadata, raw_metadata = _load_metadata(metadata_path)
    array_paths = {
        name: test_root / f"all__{name}.npy"
        for name in (
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        )
    }
    missing = [
        str(array_path)
        for array_path in array_paths.values()
        if not array_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"ARC test split is missing required arrays: {missing}")
    arrays = {
        name: np.load(array_path, mmap_mode="r")
        for name, array_path in array_paths.items()
    }
    inputs = arrays["inputs"]
    labels = arrays["labels"]
    puzzle_identifiers = arrays["puzzle_identifiers"]
    puzzle_indices = arrays["puzzle_indices"]
    group_indices = arrays["group_indices"]
    if inputs.ndim != 2 or inputs.shape[1] != metadata.seq_len or inputs.shape[0] < 1:
        raise ValueError(
            f"ARC inputs must have shape [N, {metadata.seq_len}] with N positive"
        )
    if labels.shape != inputs.shape:
        raise ValueError("ARC labels must match the input array shape")
    _validate_token_array("inputs", inputs, vocab_size=metadata.vocab_size)
    _validate_token_array("labels", labels, vocab_size=metadata.vocab_size)
    _validate_boundaries(
        "puzzle_indices", puzzle_indices, expected_stop=int(inputs.shape[0])
    )
    puzzle_count = int(puzzle_indices.size - 1)
    total_puzzles = raw_metadata.get("total_puzzles")
    if (
        isinstance(total_puzzles, bool)
        or not isinstance(total_puzzles, int)
        or total_puzzles != puzzle_count
    ):
        raise ValueError("ARC metadata total_puzzles differs from puzzle boundaries")
    if puzzle_identifiers.shape != (puzzle_count,) or not np.issubdtype(
        puzzle_identifiers.dtype, np.integer
    ):
        raise ValueError("ARC puzzle_identifiers must contain one integer per puzzle")
    identifier_values = np.asarray(puzzle_identifiers, dtype=np.int64)
    if np.any(identifier_values < 0) or np.any(
        identifier_values >= metadata.num_puzzle_identifiers
    ):
        raise ValueError("ARC puzzle identifier is outside the metadata range")
    _validate_boundaries("group_indices", group_indices, expected_stop=puzzle_count)

    identifier_payload = json.loads(identifiers_path.read_text(encoding="utf-8"))
    if (
        not isinstance(identifier_payload, list)
        or len(identifier_payload) != metadata.num_puzzle_identifiers
        or not all(isinstance(value, str) for value in identifier_payload)
    ):
        raise ValueError("ARC identifiers.json does not match metadata identifiers")
    test_puzzles = json.loads(test_puzzles_path.read_text(encoding="utf-8"))
    if not isinstance(test_puzzles, Mapping) or not test_puzzles:
        raise ValueError("ARC test_puzzles.json must contain a non-empty task object")
    _validate_test_puzzles(test_puzzles)
    identifier_tasks = {
        inverse_augmentation(identifier_payload[int(identifier_id)])[0]
        for identifier_id in np.unique(identifier_values)
    }
    if identifier_tasks != set(test_puzzles):
        raise ValueError(
            "ARC identifier tasks differ from the test_puzzles.json sidecar"
        )
    total_groups = raw_metadata.get("total_groups")
    if isinstance(total_groups, bool) or not isinstance(total_groups, int):
        raise ValueError("ARC metadata 'total_groups' must be an integer")
    if total_groups != int(group_indices.size - 1) or total_groups != len(test_puzzles):
        raise ValueError("ARC task count differs across metadata, groups, and sidecar")

    return ArcDataset(
        root=root,
        metadata=metadata,
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=puzzle_identifiers,
        puzzle_indices=puzzle_indices,
        group_indices=group_indices,
        identifier_map=tuple(identifier_payload),
        test_puzzles=test_puzzles,
    )


def _resolved_config_payload(config: ArcEvaluationConfig) -> dict[str, Any]:
    return {
        "task": config.task,
        "method": config.method,
        "checkpoint": str(config.checkpoint),
        "config": str(config.config),
        "dataset": str(config.dataset),
        "output": str(config.output),
        "candidate_count": config.candidate_count,
        "depth": config.depth,
        "global_batch_size": config.global_batch_size,
        "seed": config.seed,
        "device": config.device,
        "latent_noise_scale": config.latent_noise_scale,
        "parameter_perturbation_scale": config.parameter_perturbation_scale,
        "max_batches": config.max_batches,
    }


def _write_rank_predictions(
    output: Path, *, rank: int, predictions: Sequence[ArcPrediction]
) -> Path:
    path = output / "rank_predictions" / f"rank_{rank}_predictions.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = [
        {
            "task_name": prediction.task_name,
            "input_hash": prediction.input_hash,
            "grid": prediction.grid,
            "q_logit": prediction.q_logit,
            "row_index": prediction.row_index,
        }
        for prediction in predictions
    ]
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def _load_rank_predictions(output: Path, *, world_size: int) -> list[ArcPrediction]:
    predictions: list[ArcPrediction] = []
    for rank in range(world_size):
        path = output / "rank_predictions" / f"rank_{rank}_predictions.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Malformed ARC rank prediction file: {path}")
        for value in payload:
            if not isinstance(value, Mapping):
                raise ValueError(f"Malformed ARC rank prediction file: {path}")
            task_name = value.get("task_name")
            input_hash = value.get("input_hash")
            q_logit = value.get("q_logit")
            row_index = value.get("row_index")
            if (
                not isinstance(task_name, str)
                or not isinstance(input_hash, str)
                or not input_hash
                or isinstance(row_index, bool)
                or not isinstance(row_index, int)
            ):
                raise ValueError(f"Malformed ARC rank prediction file: {path}")
            try:
                score = float(q_logit)
                grid = _arc_grid(value.get("grid")).copy()
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Malformed ARC rank prediction file: {path}"
                ) from error
            if not math.isfinite(score):
                raise ValueError(f"Malformed ARC rank prediction file: {path}")
            predictions.append(
                ArcPrediction(
                    task_name=task_name,
                    input_hash=input_hash,
                    grid=grid,
                    q_logit=score,
                    row_index=row_index,
                )
            )
    return predictions


def _selected_arc_predictions(
    dataset: ArcDataset,
    batch: ArcBatch,
    candidate_predictions: Tensor,
    candidate_scores: Tensor,
) -> list[ArcPrediction]:
    batch_size = int(batch.inputs.shape[0])
    if candidate_predictions.ndim != 3 or candidate_predictions.shape[0] != batch_size:
        raise ValueError("ARC candidates must have shape [batch, candidates, seq_len]")
    if candidate_predictions.shape[2] != dataset.metadata.seq_len:
        raise ValueError("ARC candidate sequence length differs from the dataset")
    if candidate_scores.shape != candidate_predictions.shape[:2]:
        raise ValueError(
            "ARC candidate scores must match batch and candidate dimensions"
        )
    selected = ptrm.select_first_max(candidate_scores.float())
    rows = torch.arange(batch_size, device=selected.device)
    selected_grids = candidate_predictions[rows, selected].detach().cpu().numpy()
    selected_scores = candidate_scores[rows, selected].detach().float().cpu().numpy()
    inputs = batch.inputs.detach().cpu().numpy()
    identifier_ids = batch.puzzle_identifiers.detach().cpu().numpy()
    predictions: list[ArcPrediction] = []
    for local_index, row_index in enumerate(batch.row_indices.tolist()):
        identifier_id = int(identifier_ids[local_index])
        if not 0 <= identifier_id < len(dataset.identifier_map):
            raise ValueError(
                f"ARC identifier {identifier_id} is outside identifiers.json"
            )
        task_name, restore = inverse_augmentation(dataset.identifier_map[identifier_id])
        original_input = restore(crop_grid(inputs[local_index]))
        original_prediction = restore(crop_grid(selected_grids[local_index]))
        predictions.append(
            ArcPrediction(
                task_name=task_name,
                input_hash=grid_hash(original_input),
                grid=original_prediction,
                q_logit=float(selected_scores[local_index]),
                row_index=int(row_index),
            )
        )
    return predictions


def _arc_candidates(
    config: ArcEvaluationConfig,
    *,
    model: nn.Module,
    adapter: Any,
    batch: Mapping[str, Tensor],
    generator: torch.Generator,
    parameter_bank: ptrm.ParameterPerturbationBank | None,
) -> tuple[Tensor, Tensor]:
    if parameter_bank is None:
        assert config.latent_noise_scale is not None
        return adapter.evaluation_candidates(
            model,
            batch,
            candidate_count=config.candidate_count,
            latent_noise_sigma=config.latent_noise_scale,
            inference_depth=config.depth,
            generator=generator,
        )
    prediction_parts: list[Tensor] = []
    score_parts: list[Tensor] = []
    for candidate_index in range(config.candidate_count):
        parameter_bank.apply(candidate_index)
        predictions, scores = adapter.evaluation_candidates(
            model,
            batch,
            candidate_count=1,
            latent_noise_sigma=0.0,
            inference_depth=config.depth,
            generator=generator,
        )
        prediction_parts.append(predictions)
        score_parts.append(scores)
    return torch.cat(prediction_parts, dim=1), torch.cat(score_parts, dim=1)


def run_arc_evaluation(
    config: ArcEvaluationConfig,
    dataset: ArcDataset,
    *,
    model: nn.Module,
    adapter: Any,
    context: DistributedContext,
) -> ArcEvaluationResult | None:
    """Run one distributed ARC evaluation and aggregate official outputs."""

    if config.global_batch_size % context.world_size:
        raise ValueError("global batch size must be divisible by world size")
    if (config.latent_noise_scale is None) == (
        config.parameter_perturbation_scale is None
    ):
        raise ValueError("exactly one ARC sampling method must be selected")
    if context.primary:
        if config.output.exists() and (
            not config.output.is_dir() or any(config.output.iterdir())
        ):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output: {config.output}"
            )
        config.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            config.output / "resolved_config.json",
            _resolved_config_payload(config),
        )
    if context.world_size > 1:
        dist.barrier()

    started = time.perf_counter()
    generator = torch.Generator(device=context.device)
    generator.manual_seed(config.seed + context.rank)
    parameter_bank: ptrm.ParameterPerturbationBank | None = None
    if config.parameter_perturbation_scale is not None:
        bank_generator = torch.Generator(device=context.device)
        bank_generator.manual_seed(config.seed + 1_000_003)
        parameter_bank = ptrm.ParameterPerturbationBank.sample(
            model,
            candidate_count=config.candidate_count,
            relative_scale=config.parameter_perturbation_scale,
            generator=bank_generator,
        )

    rank_predictions: list[ArcPrediction] = []
    batch_count = 0
    try:
        for batch_index, batch in enumerate(
            iter_rank_batches(
                dataset,
                config,
                rank=context.rank,
                world_size=context.world_size,
            )
        ):
            if config.max_batches is not None and batch_index >= config.max_batches:
                break
            device_batch = {
                "inputs": batch.inputs.to(context.device),
                "labels": batch.labels.to(context.device),
                "puzzle_identifiers": batch.puzzle_identifiers.to(context.device),
            }
            candidate_predictions, candidate_scores = _arc_candidates(
                config,
                model=model,
                adapter=adapter,
                batch=device_batch,
                generator=generator,
                parameter_bank=parameter_bank,
            )
            device_arc_batch = ArcBatch(
                row_indices=batch.row_indices,
                inputs=device_batch["inputs"],
                labels=device_batch["labels"],
                puzzle_identifiers=device_batch["puzzle_identifiers"],
            )
            rank_predictions.extend(
                _selected_arc_predictions(
                    dataset,
                    device_arc_batch,
                    candidate_predictions,
                    candidate_scores,
                )
            )
            batch_count += 1
    finally:
        if parameter_bank is not None:
            parameter_bank.restore()

    _write_rank_predictions(
        config.output,
        rank=context.rank,
        predictions=rank_predictions,
    )
    atomic_write_json(
        config.output / f"rank_{context.rank}_runtime.json",
        {
            "rank": context.rank,
            "world_size": context.world_size,
            "batch_count": batch_count,
            "prediction_count": len(rank_predictions),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    if context.world_size > 1:
        dist.barrier()
    if not context.primary or config.max_batches is not None:
        return None

    predictions = _load_rank_predictions(config.output, world_size=context.world_size)
    row_indices = [prediction.row_index for prediction in predictions]
    if any(index is None for index in row_indices) or sorted(row_indices) != list(
        range(int(dataset.inputs.shape[0]))
    ):
        raise ValueError("ARC rank outputs do not cover every dataset row exactly once")
    predictions.sort(
        key=lambda prediction: (
            -1 if prediction.row_index is None else prediction.row_index
        )
    )
    result = aggregate_arc_predictions(
        predictions,
        test_puzzles=dataset.test_puzzles,
    )
    atomic_write_json(config.output / "metrics.json", result.metrics)
    atomic_write_json(config.output / "submission.json", result.submission)
    return result


def evaluate(config: ArcEvaluationConfig) -> ArcEvaluationResult | None:
    """Load one ARC run and execute it in the current torchrun process."""

    context = initialize_distributed(config.device)
    seed_all(config.seed + context.rank)
    dataset = load_arc_dataset(config.dataset)
    local_batch_size = config.global_batch_size // context.world_size
    model = load_arc_model(
        config,
        dataset,
        device=context.device,
        local_batch_size=local_batch_size,
    )
    return run_arc_evaluation(
        config,
        dataset,
        model=model,
        adapter=ptrm.ADAPTER,
        context=context,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = config_from_args(build_parser().parse_args(argv))
    try:
        result = evaluate(config)
        if result is not None:
            print(json.dumps(result.metrics, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
