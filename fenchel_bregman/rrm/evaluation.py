"""Common candidate aggregation and evaluation CLI for RRM models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import yaml

from rrm.utils import (
    atomic_torch_save,
    atomic_write_json,
    build_ivon_optimizer,
    get_model_adapter,
    load_ivon_optimizer_state,
    load_puzzle_split,
    seed_all,
    validate_checkpoint_metadata,
)


@dataclass(frozen=True)
class CandidateRecords:
    source_ids: Tensor
    exact: Tensor
    selection_scores: Tensor
    predictions: Tensor | None = None
    selected_exact_override: Tensor | None = None
    pass_at_k_exact_override: Tensor | None = None
    candidate_count_override: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    model: Literal["fprm", "ptrm", "gram"]
    task: Literal["maze-hard", "sudoku-extreme"]
    preset: Literal["paper", "sanity"]
    fb: bool
    parameter_perturbation_scale: float | None
    latent_noise_scale: float | None
    checkpoint: Path | None
    dataset: Path | None
    output: Path
    aggregate_only: tuple[Path, ...]
    candidate_count: int | None
    candidate0_q_margin: float | None
    depth: int | None
    seed: int
    batch_size: int
    limit: int | None
    num_shards: int
    shard_index: int
    device: str


def _validate_records(records: CandidateRecords, *, require_full_range: bool) -> None:
    if records.source_ids.ndim != 1:
        raise ValueError("source IDs must be a vector")
    count = records.source_ids.numel()
    if records.exact.ndim != 2 or records.selection_scores.shape != records.exact.shape:
        raise ValueError("exact and selection scores must be matching [N, K] matrices")
    if records.exact.shape[0] != count or records.exact.shape[1] < 1:
        raise ValueError("candidate records have invalid source/candidate dimensions")
    if records.exact.dtype != torch.bool:
        raise ValueError("candidate exact values must be boolean")
    if not torch.isfinite(records.selection_scores).all():
        raise ValueError("candidate selection scores must be finite")
    ids = records.source_ids.detach().cpu().to(torch.long)
    if records.source_ids.dtype == torch.bool or (
        records.source_ids.is_floating_point()
        and not torch.equal(
            records.source_ids.detach().cpu(), ids.to(records.source_ids.dtype)
        )
    ):
        raise ValueError("source IDs must contain integers")
    if torch.unique(ids).numel() != count:
        raise ValueError("source IDs must be unique")
    if require_full_range and not torch.equal(
        torch.sort(ids).values, torch.arange(count, dtype=torch.long)
    ):
        raise ValueError("source IDs must cover the complete range [0, N)")
    for name, override in (
        ("selected", records.selected_exact_override),
        ("pass_at_k", records.pass_at_k_exact_override),
    ):
        if override is not None and (
            override.shape != (count,) or override.dtype != torch.bool
        ):
            raise ValueError(f"{name} override must be a boolean [N] vector")
    if records.candidate_count_override is not None and records.candidate_count_override < 1:
        raise ValueError("candidate_count_override must be positive")


def _aggregate_records(
    records: CandidateRecords, *, require_full_range: bool
) -> dict[str, int | float | str]:
    _validate_records(records, require_full_range=require_full_range)
    count, candidates = records.exact.shape
    rows = torch.arange(count, device=records.exact.device)
    selected_indices = records.selection_scores.argmax(dim=1)
    selected = records.exact[rows, selected_indices]
    passed = records.exact.any(dim=1)
    if records.selected_exact_override is not None:
        selected = records.selected_exact_override.to(device=selected.device)
    if records.pass_at_k_exact_override is not None:
        passed = records.pass_at_k_exact_override.to(device=passed.device)
    selected_count = int(selected.sum().item())
    pass_count = int(passed.sum().item())
    candidate_count = (
        records.candidate_count_override
        if records.candidate_count_override is not None
        else candidates
    )
    return {
        "format": "RRM_METRICS_V1",
        "total": count,
        "selected_exact": selected_count,
        "pass_at_k_exact": pass_count,
        "candidate_count": int(candidate_count),
        "selected_accuracy": selected_count / count if count else 0.0,
        "pass_at_k_accuracy": pass_count / count if count else 0.0,
    }


def aggregate_candidates(records: CandidateRecords) -> dict[str, int | float | str]:
    """Select by stable first argmax and aggregate exact integer counts."""

    return _aggregate_records(records, require_full_range=True)


def apply_candidate0_q_margin(
    records: CandidateRecords, *, margin: float
) -> CandidateRecords:
    """Keep candidate 0 unless another raw Q score clears a strict margin."""

    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("candidate0 Q margin must be finite and non-negative")
    _validate_records(records, require_full_range=False)
    if records.exact.shape[1] < 2:
        raise ValueError("candidate0 Q margin requires at least two candidates")
    best_scores, best_indices = records.selection_scores.max(dim=1)
    candidate0_scores = records.selection_scores[:, 0]
    selected_indices = torch.where(
        best_scores > candidate0_scores + float(margin),
        best_indices,
        torch.zeros_like(best_indices),
    )
    rows = torch.arange(records.exact.shape[0], device=records.exact.device)
    return CandidateRecords(
        source_ids=records.source_ids,
        exact=records.exact,
        selection_scores=records.selection_scores,
        predictions=records.predictions,
        selected_exact_override=records.exact[rows, selected_indices],
        pass_at_k_exact_override=records.pass_at_k_exact_override,
        candidate_count_override=records.candidate_count_override,
    )


def save_candidate_records(path: Path, records: CandidateRecords) -> None:
    _validate_records(records, require_full_range=False)
    payload = {
        "format": "RRM_CANDIDATES_V1",
        "source_ids": records.source_ids.detach().cpu(),
        "exact": records.exact.detach().cpu(),
        "selection_scores": records.selection_scores.detach().cpu(),
        "predictions": (
            None if records.predictions is None else records.predictions.detach().cpu()
        ),
        "selected_exact_override": (
            None
            if records.selected_exact_override is None
            else records.selected_exact_override.detach().cpu()
        ),
        "pass_at_k_exact_override": (
            None
            if records.pass_at_k_exact_override is None
            else records.pass_at_k_exact_override.detach().cpu()
        ),
        "candidate_count_override": records.candidate_count_override,
    }
    atomic_torch_save(path, payload)


def load_legacy_records(path: Path) -> CandidateRecords:
    """Normalize the retained RRM candidate/outcome tensor schemas."""

    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.lower() == ".csv":
        with resolved.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"dataset_index", "exact_correct"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                "FPRM CSV must contain dataset_index and exact_correct"
            )

        def parse_bool(value: str) -> bool:
            normalized = value.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Invalid exact_correct value: {value!r}")
            return normalized == "true"

        return CandidateRecords(
            source_ids=torch.tensor(
                [int(row["dataset_index"]) for row in rows], dtype=torch.long
            ),
            exact=torch.tensor(
                [[parse_bool(row["exact_correct"])] for row in rows],
                dtype=torch.bool,
            ),
            selection_scores=torch.tensor(
                [[float(row.get("q_score") or 0.0)] for row in rows],
                dtype=torch.float32,
            ),
            candidate_count_override=1,
        )
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Candidate archive must contain a mapping: {path}")
    if payload.get("format") == "RRM_CANDIDATES_V1":
        return CandidateRecords(
            source_ids=payload["source_ids"],
            exact=payload["exact"],
            selection_scores=payload["selection_scores"],
            predictions=payload.get("predictions"),
            selected_exact_override=payload.get("selected_exact_override"),
            pass_at_k_exact_override=payload.get("pass_at_k_exact_override"),
            candidate_count_override=payload.get("candidate_count_override"),
        )
    if isinstance(payload.get("outcomes"), Mapping):
        payload = payload["outcomes"]
    source_ids = payload.get(
        "source_ids", payload.get("source_indices", payload.get("puzzle_index"))
    )
    if not isinstance(source_ids, Tensor):
        raise ValueError("Legacy candidate archive is missing source IDs")
    exact = payload.get("exact", payload.get("sample_exact"))
    scores = payload.get("selection_scores", payload.get("q_scores", payload.get("values")))
    if isinstance(exact, Tensor) and isinstance(scores, Tensor):
        if exact.ndim == 1:
            exact = exact.unsqueeze(1)
        if scores.ndim == 1:
            scores = scores.unsqueeze(1)
        return CandidateRecords(source_ids=source_ids, exact=exact.bool(), selection_scores=scores.float())
    selected = payload.get(
        "best_q_exact",
        payload.get("selected_exact", payload.get("lprm_exact")),
    )
    passed = payload.get("pass_at_k_exact", payload.get("pass_exact"))
    direct = payload.get("direct_exact")
    if not isinstance(selected, Tensor):
        raise ValueError("Legacy candidate archive is missing selected outcomes")
    if not isinstance(passed, Tensor):
        passed = selected
    if not isinstance(direct, Tensor):
        direct = selected
    metadata = payload.get("metadata")
    metadata_k = metadata.get("k", 1) if isinstance(metadata, Mapping) else 1
    legacy_config = payload.get("config")
    if isinstance(legacy_config, Mapping):
        metadata_k = legacy_config.get("k", metadata_k)
    if "exact_candidates" in payload and metadata_k == 1:
        # The retained GRAM full-evaluation bank predates embedded metadata and
        # was frozen with ten candidates per source.
        metadata_k = 10
    candidate_count = int(
        payload.get("candidate_count", payload.get("k", metadata_k))
    )
    return CandidateRecords(
        source_ids=source_ids,
        exact=direct.bool().reshape(-1, 1),
        selection_scores=torch.zeros(source_ids.numel(), 1),
        selected_exact_override=selected.bool().reshape(-1),
        pass_at_k_exact_override=passed.bool().reshape(-1),
        candidate_count_override=candidate_count,
    )


def _concatenate_records(records: Sequence[CandidateRecords]) -> CandidateRecords:
    if not records:
        raise ValueError("At least one candidate archive is required")
    for record in records:
        _validate_records(record, require_full_range=False)
    widths = {record.exact.shape[1] for record in records}
    if len(widths) != 1:
        raise ValueError("Candidate archives use different candidate widths")
    override_counts = {
        record.candidate_count_override
        if record.candidate_count_override is not None
        else record.exact.shape[1]
        for record in records
    }
    if len(override_counts) != 1:
        raise ValueError("Candidate archives use different candidate counts")
    order = torch.argsort(torch.cat([record.source_ids.long() for record in records]))

    def concatenate(name: str) -> Tensor | None:
        values = [getattr(record, name) for record in records]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Candidate shards differ on optional field {name}")
        return torch.cat(values)[order]  # type: ignore[arg-type]

    return CandidateRecords(
        source_ids=torch.cat([record.source_ids for record in records])[order],
        exact=torch.cat([record.exact for record in records])[order],
        selection_scores=torch.cat([record.selection_scores for record in records])[order],
        predictions=concatenate("predictions"),
        selected_exact_override=concatenate("selected_exact_override"),
        pass_at_k_exact_override=concatenate("pass_at_k_exact_override"),
        candidate_count_override=override_counts.pop(),
    )


def aggregate_candidate_files(paths: Sequence[Path]) -> dict[str, int | float | str]:
    loaded = [load_legacy_records(path) for path in paths]
    if len(loaded) > 1 and all(
        torch.equal(record.source_ids, loaded[0].source_ids) for record in loaded[1:]
    ):
        if any(record.exact.shape[1] != 1 for record in loaded):
            raise ValueError("Repeated-source archives must contain one draw each")
        records = CandidateRecords(
            source_ids=loaded[0].source_ids,
            exact=torch.cat([record.exact for record in loaded], dim=1),
            selection_scores=torch.cat(
                [record.selection_scores for record in loaded], dim=1
            ),
            candidate_count_override=len(loaded),
        )
    else:
        records = _concatenate_records(loaded)
    return aggregate_candidates(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate or re-aggregate RRM candidate predictions."
    )
    parser.add_argument("--model", required=True, choices=("fprm", "ptrm", "gram"))
    parser.add_argument("--task", required=True, choices=("maze-hard", "sudoku-extreme"))
    parser.add_argument("--preset", choices=("paper", "sanity"), default="paper")
    parser.add_argument("--fb", action="store_true")
    parser.add_argument("--parameter-perturbation-scale", type=float)
    parser.add_argument("--latent-noise-scale", type=float)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aggregate-only", type=Path, nargs="+", default=())
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--candidate0-q-margin", type=float)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    if args.seed < 0 or args.batch_size < 1:
        raise ValueError("seed must be non-negative and batch_size positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.preset == "paper" and args.seed != 0:
        raise ValueError(f"paper preset expected seed=0, actual seed={args.seed}")
    limit = args.limit
    candidates = args.candidate_count
    if args.preset == "sanity":
        limit = 8 if limit is None else min(int(limit), 8)
        candidates = 1
    if args.aggregate_only:
        if args.checkpoint is not None or args.dataset is not None:
            raise ValueError("aggregate-only must not load checkpoint or dataset")
    elif args.checkpoint is None or args.dataset is None:
        raise ValueError("live evaluation requires checkpoint and dataset")
    if args.parameter_perturbation_scale is not None and args.parameter_perturbation_scale < 0:
        raise ValueError("parameter perturbation scale must be non-negative")
    if args.fb and args.parameter_perturbation_scale is not None:
        raise ValueError("--fb and --parameter-perturbation-scale are mutually exclusive")
    if args.latent_noise_scale is not None and args.latent_noise_scale < 0:
        raise ValueError("latent noise scale must be non-negative")
    if args.candidate0_q_margin is not None:
        if (
            not math.isfinite(float(args.candidate0_q_margin))
            or args.candidate0_q_margin < 0
        ):
            raise ValueError("candidate0 Q margin must be finite and non-negative")
        if args.model != "ptrm":
            raise ValueError("candidate0 Q margin is only supported for PTRM")
        if candidates == 1:
            raise ValueError("candidate0 Q margin requires multiple candidates")
    return EvaluationConfig(
        model=args.model,
        task=args.task,
        preset=args.preset,
        fb=bool(args.fb),
        parameter_perturbation_scale=args.parameter_perturbation_scale,
        latent_noise_scale=args.latent_noise_scale,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        output=args.output,
        aggregate_only=tuple(args.aggregate_only),
        candidate_count=candidates,
        candidate0_q_margin=args.candidate0_q_margin,
        depth=args.depth,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        limit=limit,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        device=str(args.device),
    )


def _state_mapping(payload: Mapping[str, Any], model: str) -> Mapping[str, Tensor]:
    if payload.get("format") == "RRM_CHECKPOINT_V1":
        state = payload.get("model_state")
    elif model == "gram":
        state = payload.get("ema_model_state_dict")
    elif model == "ptrm" and isinstance(payload.get("model_state_dict"), Mapping):
        state = payload.get("model_state_dict")
    elif isinstance(payload.get("model"), Mapping):
        state = payload.get("model")
    else:
        state = payload
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Checkpoint does not contain a tensor model state")
    return state  # type: ignore[return-value]


def _find_state_tensor(state: Mapping[str, Tensor], suffix: str) -> Tensor:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint tensor ending in {suffix!r}")
    return matches[0]


def _evaluation_model_metadata(
    config: EvaluationConfig,
    payload: Mapping[str, Any],
    *,
    sequence_length: int,
) -> dict[str, Any]:
    if payload.get("format") == "RRM_CHECKPOINT_V1":
        validate_checkpoint_metadata(
            payload,
            model=config.model,
            task=config.task,
            fb=config.fb,
            preset=config.preset,
        )
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, Mapping) or not isinstance(
            resolved.get("model_metadata"), Mapping
        ):
            raise ValueError("RRM checkpoint lacks resolved model metadata")
        metadata = copy.deepcopy(dict(resolved["model_metadata"]))
    elif config.model == "gram":
        raw = payload.get("config")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
            raise ValueError("GRAM checkpoint lacks model config")
        model_values = dict(raw["model"])
        model_values.setdefault("positionwise_initial_state", False)
        model_values.setdefault("share_recursive_core", True)
        metadata = {"model": model_values}
    elif config.model == "fprm" and isinstance(payload.get("registered_config"), Mapping):
        registered = payload["registered_config"]
        pretrain = registered.get("pretrain_config")
        if not isinstance(pretrain, Mapping) or not isinstance(pretrain.get("arch"), Mapping):
            raise ValueError("FPRM posterior lacks registered architecture")
        metadata = {"arch": copy.deepcopy(dict(pretrain["arch"]))}
    elif config.model == "ptrm" and isinstance(payload.get("args"), Mapping):
        args = payload["args"]
        names = (
            "H_cycles",
            "L_cycles",
            "H_layers",
            "L_layers",
            "hidden_size",
            "expansion",
            "num_heads",
            "pos_encodings",
            "halt_max_steps",
            "halt_exploration_prob",
            "forward_dtype",
            "mlp_t",
            "puzzle_emb_len",
            "no_ACT_continue",
            "puzzle_emb_ndim",
        )
        metadata = {
            "arch": {
                **{name: args[name] for name in names},
                "loss": {"q_loss_coeff": float(args.get("q_loss_weight", 0.5))},
            }
        }
    else:
        if config.checkpoint is None:
            raise ValueError("checkpoint path is required")
        config_path = config.checkpoint.parent / "all_config.yaml"
        if not config_path.is_file():
            raise ValueError("Released checkpoint is missing sibling all_config.yaml")
        released = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(released, Mapping) or not isinstance(released.get("arch"), Mapping):
            raise ValueError("Released config lacks architecture")
        metadata = {"arch": copy.deepcopy(dict(released["arch"]))}
    candidates = config.candidate_count or (
        1 if config.model == "fprm" else 10
    )
    depth = config.depth or (
        64 if config.task == "sudoku-extreme" else 16
    )
    if config.model in ("fprm", "ptrm"):
        state = _state_mapping(payload, config.model)
        arch = dict(metadata["arch"])
        if config.model == "fprm" and config.preset == "paper":
            arch.update(
                max_iter_eval=35_000,
                stepsize_decay_eval=(
                    0.997 if config.task == "sudoku-extreme" else 0.996
                ),
                decay_patience=10,
            )
        arch.update(
            batch_size=config.batch_size,
            seq_len=sequence_length,
            vocab_size=_find_state_tensor(
                state, "inner.embed_tokens.embedding_weight"
            ).shape[0],
            num_puzzle_identifiers=_find_state_tensor(
                state, "inner.puzzle_emb.weights"
            ).shape[0],
        )
        metadata["arch"] = arch
    metadata.update(
        candidate_count=candidates,
        inference_depth=depth,
        supervision_steps=depth,
        latent_noise_sigma=(
            config.latent_noise_scale
            if config.latent_noise_scale is not None
            else (0.3 if config.task == "sudoku-extreme" else 1.0)
        ),
    )
    return metadata


def _ivon_from_checkpoint(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
) -> torch.optim.Optimizer:
    if payload.get("format") == "RRM_CHECKPOINT_V1":
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, Mapping) or not isinstance(resolved.get("runtime"), Mapping):
            raise ValueError("RRM posterior checkpoint lacks optimizer runtime config")
        preset = dict(resolved["runtime"])
        state = payload.get("optimizer_state")
        step = payload.get("step")
    elif isinstance(payload.get("registered_config"), Mapping):
        registered = payload["registered_config"]
        preset = {
            "learning_rate": registered["learning_rate"],
            "beta1": 0.9,
            "ivon_beta2": 0.999,
            "ivon_ess": 100_000.0,
            "ivon_hess_init": 1.0,
            "ivon_weight_decay": 1e-4,
            "ivon_mc_samples": 1,
            "ivon_hess_approx": "price",
        }
        state = payload.get("optimizer")
        step = payload.get("ivon_current_step", payload.get("step"))
    elif isinstance(payload.get("args"), Mapping):
        args = payload["args"]
        preset = {
            "learning_rate": args["lr"],
            "beta1": args["beta1"],
            "ivon_beta2": args["ivon_beta2"],
            "ivon_ess": args["ivon_ess"],
            "ivon_hess_init": args["ivon_hess_init"],
            "ivon_weight_decay": args["ivon_weight_decay"],
            "ivon_mc_samples": args["ivon_mc_samples"],
            "ivon_hess_approx": args["ivon_hess_approx"],
        }
        state = payload.get("optimizer_state_dict")
        step = payload.get("dense_optimizer_current_step", payload.get("step"))
    else:
        raise ValueError("Posterior checkpoint lacks an IVON contract")
    if not isinstance(state, Mapping) or not isinstance(step, int):
        raise ValueError("Posterior checkpoint lacks IVON state/step")
    optimizer = build_ivon_optimizer(model, preset)
    load_ivon_optimizer_state(optimizer, state, current_step=step)
    return optimizer


def _parameter_mean_abs(model: torch.nn.Module) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in parameters)
    if not count:
        raise ValueError("model has no trainable parameters")
    return sum(float(parameter.detach().float().abs().sum()) for parameter in parameters) / count


@contextmanager
def _perturbed_parameters(
    model: torch.nn.Module,
    *,
    sigma: float,
    generator: torch.Generator,
) -> Iterator[None]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    originals = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
                noise = torch.randn(
                    parameter.shape,
                    device=parameter.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                parameter.add_(noise.to(parameter.dtype), alpha=sigma)
        yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)


def _point_candidates(
    adapter: Any,
    model: torch.nn.Module,
    batch: Mapping[str, Tensor],
    config: EvaluationConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if config.model == "ptrm":
        return adapter.evaluation_candidates(
            model,
            batch,
            candidate_count=1,
            latent_noise_sigma=0.0,
            inference_depth=config.depth or (
                64 if config.task == "sudoku-extreme" else 16
            ),
            generator=generator,
        )
    if config.model == "gram":
        return adapter.evaluation_candidates(
            model,
            batch,
            candidate_count=1,
            supervision_steps=config.depth,
            generator=generator,
        )
    output = adapter.evaluation_forward(model, batch)
    return output.predictions.unsqueeze(1), output.selection_score.unsqueeze(1)


def _evaluation_batch(
    split: Any,
    start: int,
    stop: int,
    device: torch.device,
    *,
    pad_to: int | None = None,
) -> dict[str, Tensor]:
    if not 0 <= start < stop <= len(split.inputs):
        raise ValueError("evaluation batch bounds are outside the puzzle split")
    inputs = np.asarray(split.inputs[start:stop], dtype=np.int64).copy()
    labels = np.asarray(split.labels[start:stop], dtype=np.int64).copy()
    puzzle_identifiers = np.asarray(
        split.puzzle_identifiers[start:stop], dtype=np.int64
    ).copy()
    if split.ignore_label_id is not None:
        labels[labels == int(split.ignore_label_id)] = -100
    if pad_to is not None:
        if pad_to < len(inputs):
            raise ValueError("pad_to cannot be smaller than the evaluation batch")
        padding = pad_to - len(inputs)
        if padding:
            inputs = np.pad(
                inputs,
                ((0, padding), (0, 0)),
                constant_values=int(split.pad_id),
            )
            labels = np.pad(
                labels,
                ((0, padding), (0, 0)),
                constant_values=-100,
            )
            puzzle_identifiers = np.pad(
                puzzle_identifiers,
                (0, padding),
                constant_values=int(split.blank_identifier_id),
            )
    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "puzzle_identifiers": torch.from_numpy(puzzle_identifiers).to(device),
    }


@torch.inference_mode()
def _evaluate_live(config: EvaluationConfig) -> CandidateRecords:
    assert config.checkpoint is not None and config.dataset is not None
    effective_seed = (
        config.seed + config.shard_index
        if config.model == "gram" and not config.fb
        else config.seed
    )
    seed_all(effective_seed)
    split = load_puzzle_split(config.dataset, split="test", task=config.task)
    full_total = (
        len(split.inputs)
        if config.limit is None
        else min(len(split.inputs), config.limit)
    )
    shard_start = full_total * config.shard_index // config.num_shards
    shard_stop = full_total * (config.shard_index + 1) // config.num_shards
    payload = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation checkpoint must contain a mapping")
    metadata = _evaluation_model_metadata(
        config, payload, sequence_length=int(split.inputs.shape[1])
    )
    adapter = get_model_adapter(config.model)
    adapter_preset = (
        "trm"
        if config.model == "ptrm" and int(metadata["candidate_count"]) == 1
        else config.model
    )
    model = adapter.build_model(config.task, adapter_preset, metadata)
    device = torch.device(config.device)
    model.to(device).eval()
    adapter.load_checkpoint(model, config.checkpoint)
    candidates = int(metadata["candidate_count"])
    posterior = _ivon_from_checkpoint(model, payload) if config.fb else None
    if posterior is not None:
        # The verified posterior evaluator resets the sampling stream after
        # model/optimizer construction so initialization cannot consume draws.
        seed_all(effective_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(effective_seed)
    if config.model == "fprm" and candidates > 1 and (
        posterior is not None or config.parameter_perturbation_scale is not None
    ):
        row_count = shard_stop - shard_start
        exact = torch.empty(row_count, candidates, dtype=torch.bool)
        scores = torch.empty(row_count, candidates, dtype=torch.float32)
        normalized_sigma = (
            None
            if config.parameter_perturbation_scale is None
            else config.parameter_perturbation_scale * _parameter_mean_abs(model)
        )
        for candidate_index in range(candidates):
            if posterior is not None:
                seed_all(62_000_000 + candidate_index)
                context = posterior.sampled_params(train=False)  # type: ignore[attr-defined]
            else:
                assert normalized_sigma is not None
                draw_generator = torch.Generator(device=device)
                draw_generator.manual_seed(1_000_003 + candidate_index)
                context = _perturbed_parameters(
                    model, sigma=normalized_sigma, generator=draw_generator
                )
            with context:
                for start in range(shard_start, shard_stop, config.batch_size):
                    stop = min(start + config.batch_size, shard_stop)
                    effective = stop - start
                    batch = _evaluation_batch(
                        split,
                        start,
                        stop,
                        device,
                        pad_to=config.batch_size,
                    )
                    predictions, draw_scores = _point_candidates(
                        adapter, model, batch, config, generator
                    )
                    labels = batch["labels"][:effective]
                    valid = labels != -100
                    offset = start - shard_start
                    exact[offset : offset + effective, candidate_index] = (
                        ((predictions[:effective, 0] == labels) | ~valid)
                        .all(dim=-1)
                        .cpu()
                    )
                    scores[offset : offset + effective, candidate_index] = (
                        draw_scores[:effective, 0].float().cpu()
                    )
        return CandidateRecords(
            source_ids=torch.arange(shard_start, shard_stop, dtype=torch.long),
            exact=exact,
            selection_scores=scores,
            candidate_count_override=candidates,
        )
    exact_rows: list[Tensor] = []
    score_rows: list[Tensor] = []
    source_rows: list[Tensor] = []
    for start in range(shard_start, shard_stop, config.batch_size):
        stop = min(start + config.batch_size, shard_stop)
        effective = stop - start
        batch = _evaluation_batch(
            split,
            start,
            stop,
            device,
            pad_to=config.batch_size if config.model == "fprm" else None,
        )
        if posterior is not None or config.parameter_perturbation_scale is not None:
            prediction_parts: list[Tensor] = []
            score_parts: list[Tensor] = []
            normalized_sigma = (
                None
                if config.parameter_perturbation_scale is None
                else config.parameter_perturbation_scale * _parameter_mean_abs(model)
            )
            for _ in range(candidates):
                if posterior is not None:
                    context = posterior.sampled_params(train=False)  # type: ignore[attr-defined]
                else:
                    assert normalized_sigma is not None
                    context = _perturbed_parameters(
                        model, sigma=normalized_sigma, generator=generator
                    )
                with context:
                    predictions, scores = _point_candidates(
                        adapter, model, batch, config, generator
                    )
                prediction_parts.append(predictions)
                score_parts.append(scores)
            predictions = torch.cat(prediction_parts, dim=1)
            scores = torch.cat(score_parts, dim=1)
        elif config.model == "ptrm":
            predictions, scores = adapter.evaluation_candidates(
                model,
                batch,
                candidate_count=candidates,
                latent_noise_sigma=float(metadata["latent_noise_sigma"]),
                inference_depth=int(metadata["inference_depth"]),
                generator=generator,
            )
        elif config.model == "gram":
            predictions, scores = adapter.evaluation_candidates(
                model,
                batch,
                candidate_count=candidates,
                supervision_steps=int(metadata["supervision_steps"]),
                generator=generator,
            )
        else:
            predictions, scores = _point_candidates(
                adapter, model, batch, config, generator
            )
        predictions = predictions[:effective]
        scores = scores[:effective]
        labels = batch["labels"][:effective].unsqueeze(1)
        valid = labels != -100
        exact = ((predictions == labels) | ~valid).all(dim=-1)
        exact_rows.append(exact.cpu())
        score_rows.append(scores.float().cpu())
        source_rows.append(torch.arange(start, stop, dtype=torch.long))
    return CandidateRecords(
        source_ids=torch.cat(source_rows),
        exact=torch.cat(exact_rows),
        selection_scores=torch.cat(score_rows),
        candidate_count_override=candidates,
    )


def run_evaluation(config: EvaluationConfig) -> Path:
    if config.output.exists() and any(config.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config.output / "resolved_config.json",
        {
            "model": config.model,
            "task": config.task,
            "preset": config.preset,
            "fb": config.fb,
            "parameter_perturbation_scale": config.parameter_perturbation_scale,
            "latent_noise_scale": config.latent_noise_scale,
            "checkpoint": None if config.checkpoint is None else str(config.checkpoint),
            "dataset": None if config.dataset is None else str(config.dataset),
            "candidate_count": config.candidate_count,
            "candidate0_q_margin": config.candidate0_q_margin,
            "depth": config.depth,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "limit": config.limit,
            "num_shards": config.num_shards,
            "shard_index": config.shard_index,
        },
    )
    records = (
        _concatenate_records(
            [load_legacy_records(path) for path in config.aggregate_only]
        )
        if config.aggregate_only
        else _evaluate_live(config)
    )
    if config.candidate0_q_margin is not None:
        records = apply_candidate0_q_margin(
            records, margin=config.candidate0_q_margin
        )
    outcomes = config.output / "outcomes.pt"
    save_candidate_records(outcomes, records)
    metrics = _aggregate_records(
        records,
        require_full_range=bool(config.aggregate_only) or config.num_shards == 1,
    )
    atomic_write_json(config.output / "metrics.json", metrics)
    return config.output / "metrics.json"


def main(argv: Sequence[str] | None = None) -> None:
    config = config_from_args(build_parser().parse_args(argv))
    print(run_evaluation(config))


if __name__ == "__main__":
    main()
