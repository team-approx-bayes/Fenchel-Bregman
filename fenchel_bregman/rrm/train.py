"""Common training entry point for recursive reasoning models."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.optim import Optimizer

from rrm.fenchel_bregman import (
    FenchelBregmanConfig,
    FailureProbabilityTracker,
    fenchel_bregman_objective,
    price_anchored_curvature_gradient,
    should_refresh_curvature,
)
from rrm.utils import (
    DistributedContext,
    TrainingOutput,
    PuzzleTrainStream,
    atomic_torch_save,
    atomic_write_json,
    build_ivon_optimizer,
    capture_rng_state,
    get_model_adapter,
    gather_rank_ordered_vectors,
    initialize_distributed,
    load_ivon_optimizer_state,
    restore_rng_state,
    seed_all,
    sha256_file,
    sum_distributed_gradients,
    validate_checkpoint_metadata,
)


ModelName = Literal["fprm", "ptrm", "gram"]
TaskName = Literal["maze-hard", "sudoku-extreme"]
PresetName = Literal["paper", "sanity"]


@dataclass(frozen=True)
class TrainConfig:
    model: ModelName
    task: TaskName
    preset: PresetName
    fb: bool
    seed: int
    dataset: Path
    checkpoint: Path | None
    output: Path
    max_updates: int
    device: str


_PAPER_UPDATES: dict[tuple[str, str, bool], int] = {
    ("fprm", "maze-hard", True): 2000,
    ("ptrm", "maze-hard", True): 188,
    ("ptrm", "sudoku-extreme", True): 200,
    ("gram", "maze-hard", False): 500,
    ("gram", "sudoku-extreme", False): 65104,
}


_GRAM_PAPER_METADATA: dict[str, dict[str, Any]] = {
    "maze-hard": {
        "model": {
            "sequence_length": 900,
            "vocab_size": 6,
            "hidden_size": 512,
            "register_tokens": 16,
            "core_layers": 2,
            "low_level_steps": 4,
            "transitions_per_supervision": 3,
            "supervision_steps": 16,
            "core_expansion": 4.0,
            "guidance_expansion": 1.0,
            "decoder_expansion": 4.0,
            "width_multiple": 256,
            "rms_norm_eps": 1e-5,
            "minimum_std": 1e-4,
            "maximum_std": 0.01,
            "forward_dtype": "bfloat16",
            "core_architecture": "attention",
            "attention_heads": 8,
            "position_encoding": "rope",
            "puzzle_embedding_tokens": 16,
            "share_recursive_core": True,
            "positionwise_initial_state": False,
            "decoder_type": "identity",
            "guidance_mode": "zero",
            "rope_style": "half",
        },
        "kl_beta": 1e-8,
        "kl_balance": 0.8,
        "halt_loss_weight": 1e-8,
        "lprm_loss_weight": 1e-8,
        "route_token_loss_weight": 1.0,
        "route_auxiliary_loss_weight": 0.0,
        "route_auxiliary_pos_weight": 1.0,
        "route_connectivity_loss_weight": 0.0,
        "token_loss_type": "stablemax_cross_entropy",
        "kl_reduction": "sum_hidden_mean_tokens",
        "latent_source": "prior",
        "posterior_fusion": "global_token_mixer",
    },
    "sudoku-extreme": {
        "model": {
            "sequence_length": 81,
            "vocab_size": 11,
            "hidden_size": 512,
            "register_tokens": 16,
            "core_layers": 2,
            "low_level_steps": 6,
            "transitions_per_supervision": 3,
            "supervision_steps": 16,
            "core_expansion": 4.0,
            "guidance_expansion": 1.0,
            "decoder_expansion": 4.0,
            "width_multiple": 256,
            "rms_norm_eps": 1e-5,
            "minimum_std": 1e-4,
            "maximum_std": None,
            "forward_dtype": "bfloat16",
        },
        "kl_beta": 0.1,
        "kl_balance": 0.8,
        "halt_loss_weight": 0.5,
        "lprm_loss_weight": 1.0,
        "route_token_loss_weight": 1.0,
        "route_auxiliary_loss_weight": 0.0,
        "route_auxiliary_pos_weight": 1.0,
        "route_connectivity_loss_weight": 0.0,
        "token_loss_type": "stablemax_cross_entropy",
        "kl_reduction": "sum_hidden_mean_tokens",
        "latent_source": "posterior",
        "posterior_fusion": "global_token_mixer",
    },
}


_PTRM_MAZE_PAPER_ARCH: dict[str, Any] = {
    "H_cycles": 3,
    "L_cycles": 4,
    "H_layers": 0,
    "L_layers": 2,
    "hidden_size": 512,
    "expansion": 4.0,
    "num_heads": 8,
    "pos_encodings": "rope",
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "forward_dtype": "bfloat16",
    "mlp_t": False,
    "puzzle_emb_len": 16,
    "no_ACT_continue": True,
    "puzzle_emb_ndim": 512,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FPRM, PTRM/TRM, or GRAM with an optional FB objective."
    )
    parser.add_argument("--model", required=True, choices=("fprm", "ptrm", "gram"))
    parser.add_argument(
        "--task", required=True, choices=("maze-hard", "sudoku-extreme")
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preset", choices=("paper", "sanity"), default="paper")
    parser.add_argument("--fb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def resolve_train_preset(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.preset == "paper":
        if args.seed != 0:
            raise ValueError(
                f"paper preset expected seed=0, actual seed={args.seed}"
            )
        key = (args.model, args.task, bool(args.fb))
        if key not in _PAPER_UPDATES:
            raise ValueError(
                "No frozen paper training preset for "
                f"model={args.model}, task={args.task}, fb={bool(args.fb)}"
            )
        expected_updates = _PAPER_UPDATES[key]
        if args.max_updates is not None and args.max_updates != expected_updates:
            raise ValueError(
                "paper preset max_updates mismatch: "
                f"expected {expected_updates}, actual {args.max_updates}"
            )
        return {
            "max_updates": expected_updates,
            "seed": 0,
            "single_seed": True,
        }
    maximum = 1 if args.max_updates is None else args.max_updates
    if maximum < 1:
        raise ValueError("max_updates must be positive")
    if maximum > 1:
        raise ValueError("sanity preset permits at most one update")
    return {"max_updates": maximum, "seed": int(args.seed), "single_seed": True}


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    resolved = resolve_train_preset(args)
    return TrainConfig(
        model=args.model,
        task=args.task,
        preset=args.preset,
        fb=bool(args.fb),
        seed=int(args.seed),
        dataset=Path(args.dataset),
        checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        output=Path(args.output),
        max_updates=int(resolved["max_updates"]),
        device=str(args.device),
    )


def validate_resume_checkpoint(
    payload: Mapping[str, Any], config: TrainConfig
) -> None:
    validate_checkpoint_metadata(
        payload,
        model=config.model,
        task=config.task,
        fb=config.fb,
        preset=config.preset,
    )


def compute_training_loss(
    output: TrainingOutput,
    *,
    fb_config: FenchelBregmanConfig | None,
    tracker: FailureProbabilityTracker | None,
    group_ids: Tensor,
    update_tracker: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    task_loss = output.per_example_task_loss
    if task_loss.ndim != 1 or task_loss.shape != group_ids.shape:
        raise ValueError("task loss and group IDs must be matching vectors")
    metrics = dict(output.metrics)
    if fb_config is None:
        if tracker is not None:
            raise ValueError("baseline loss must not receive a failure tracker")
        metrics["task_loss"] = task_loss.detach()
        return task_loss.sum() + output.auxiliary_loss, metrics
    if tracker is None:
        raise ValueError("Fenchel--Bregman loss requires a failure tracker")
    if tracker.beta != fb_config.tracker_beta:
        raise ValueError("Fenchel--Bregman tracker beta differs from config")
    failure_probability = tracker.values_for(group_ids)
    fb = fenchel_bregman_objective(
        task_loss, failure_probability, config=fb_config
    )
    if update_tracker:
        tracker.update(
            group_ids,
            fb.failure,
            duplicate_update=fb_config.duplicate_update,
        )
    metrics.update(
        task_loss=task_loss.detach(),
        fb_failure=fb.failure.detach(),
        fb_weight=fb.weight.detach(),
        fb_loss=fb.loss.detach(),
    )
    return fb.loss + output.auxiliary_loss, metrics


def build_checkpoint(
    *,
    config: TrainConfig,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    tracker: FailureProbabilityTracker | None,
    resolved_config: Mapping[str, Any],
    source_checkpoint_sha256: str | None,
    adapter_state: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(step, bool) or int(step) != step or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {
        "format": "RRM_CHECKPOINT_V1",
        "model": config.model,
        "task": config.task,
        "fb": config.fb,
        "step": int(step),
        "seed": config.seed,
        "preset": config.preset,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "tracker_state": (
            None if tracker is None else copy.deepcopy(tracker.state_dict())
        ),
        "resolved_config": copy.deepcopy(dict(resolved_config)),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "rng_state": copy.deepcopy(capture_rng_state()),
        "adapter_state": copy.deepcopy(dict(adapter_state)),
    }


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    config: TrainConfig,
    model: nn.Module,
    optimizer: Optimizer,
    tracker: FailureProbabilityTracker | None,
) -> tuple[int, dict[str, Any]]:
    validate_resume_checkpoint(payload, config)
    required = {
        "step",
        "seed",
        "model_state",
        "optimizer_state",
        "tracker_state",
        "resolved_config",
        "source_checkpoint_sha256",
        "rng_state",
        "adapter_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"training checkpoint is missing keys: {sorted(missing)}")
    if type(payload["seed"]) is not int or payload["seed"] != config.seed:
        raise ValueError(
            f"checkpoint seed mismatch: expected {config.seed}, got {payload['seed']!r}"
        )
    model_state = payload["model_state"]
    optimizer_state = payload["optimizer_state"]
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint model/optimizer state must be mappings")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    tracker_state = payload["tracker_state"]
    if tracker is None:
        if tracker_state is not None:
            raise ValueError("baseline resume checkpoint unexpectedly contains tracker state")
    else:
        if not isinstance(tracker_state, Mapping):
            raise ValueError("FB resume checkpoint is missing tracker state")
        tracker.load_state_dict(tracker_state)
    rng_state = payload["rng_state"]
    adapter_state = payload["adapter_state"]
    if isinstance(rng_state, Mapping) and rng_state.get("format") == "RRM_DISTRIBUTED_STATE_V1":
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rng_state.get("world_size") != world_size:
            raise ValueError("checkpoint distributed world size mismatch")
        per_rank_rng = rng_state.get("per_rank")
        if not isinstance(per_rank_rng, Sequence) or len(per_rank_rng) != world_size:
            raise ValueError("checkpoint distributed RNG state is incomplete")
        rng_state = per_rank_rng[rank]
    if isinstance(adapter_state, Mapping) and adapter_state.get("format") == "RRM_DISTRIBUTED_STATE_V1":
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if adapter_state.get("world_size") != world_size:
            raise ValueError("checkpoint distributed adapter world size mismatch")
        per_rank_adapter = adapter_state.get("per_rank")
        if not isinstance(per_rank_adapter, Sequence) or len(per_rank_adapter) != world_size:
            raise ValueError("checkpoint distributed adapter state is incomplete")
        adapter_state = per_rank_adapter[rank]
    if not isinstance(rng_state, Mapping) or not isinstance(adapter_state, Mapping):
        raise ValueError("checkpoint RNG/adapter state must be mappings")
    restore_rng_state(rng_state)
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return step, copy.deepcopy(dict(adapter_state))


def _runtime_preset(config: TrainConfig) -> dict[str, Any]:
    if config.preset == "sanity":
        return {
            "batch_size": 2,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
        }
    key = (config.model, config.task, config.fb)
    presets: dict[tuple[str, str, bool], dict[str, Any]] = {
        ("fprm", "maze-hard", True): {
            "batch_size": 768,
            "train_group_count_for_epochs": 1000,
            "world_size": 8,
            "optimizer": "ivon",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "ivon_ess": 100_000.0,
            "ivon_hess_init": 1.0,
            "ivon_beta2": 0.999,
            "ivon_weight_decay": 1e-4,
            "ivon_mc_samples": 1,
            "ivon_hess_approx": "price",
            "puzzle_embedding_learning_rate": 0.01,
            "puzzle_embedding_weight_decay": 1.0,
        },
        ("ptrm", "sudoku-extreme", True): {
            "batch_size": 128,
            "train_group_count_for_epochs": 1000,
            "world_size": 1,
            "optimizer": "ivon",
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "ivon_ess": 30_000.0,
            "ivon_hess_init": 3.0,
            "ivon_beta2": 0.99999,
            "ivon_weight_decay": 0.225,
            "ivon_mc_samples": 1,
            "ivon_hess_approx": "price",
            "puzzle_embedding_learning_rate": 0.01,
            "puzzle_embedding_weight_decay": 0.1,
            "q_loss_weight": 0.35,
            "ema_decay": 0.999,
        },
        ("ptrm", "maze-hard", True): {
            "batch_size": 128,
            "train_group_count_for_epochs": 1000,
            "world_size": 1,
            "optimizer": "ivon",
            "learning_rate": 2.8565218228666595e-5,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "ivon_ess": 30_000.0,
            "ivon_hess_init": 3.0,
            "ivon_beta2": 0.99999,
            "ivon_weight_decay": 0.28734013042078377,
            "ivon_mc_samples": 1,
            "ivon_hess_approx": "price",
            "puzzle_embedding_learning_rate": 0.00013323427696276927,
            "puzzle_embedding_weight_decay": 0.1,
            "q_loss_weight": 0.27759867393254695,
            "cellwise_q_loss": True,
            "success_loss_aggregation": "smooth_weakest",
            "smooth_weakest_delta": math.log(2.0),
            "ema_decay": 0.999,
        },
        ("gram", "maze-hard", False): {
            "batch_size": 128,
            "train_group_count_for_epochs": 1000,
            "micro_batch_size": 32,
            "optimizer": "adamw",
            "learning_rate": 3e-6,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "ema_decay": 0.999,
            "gradient_clip": 1.0,
        },
        ("gram", "sudoku-extreme", False): {
            "batch_size": 768,
            "train_group_count_for_epochs": 1000,
            "micro_batch_size": 768,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "ema_decay": 0.9999,
            "learning_rate_warmup_steps": 2000,
            "gradient_clip": 1.0,
        },
    }
    try:
        return dict(presets[key])
    except KeyError as error:
        raise ValueError(f"No training runtime preset for {key}") from error


def _sanity_model_metadata(
    config: TrainConfig, stream: PuzzleTrainStream
) -> dict[str, Any]:
    sequence_length = int(stream.inputs.shape[1])
    vocab_size = int(max(stream.inputs.max(), stream.labels.max())) + 1
    identifiers = int(stream.metadata.get("num_puzzle_identifiers", 1))
    if config.model == "gram":
        return {
            "model": {
                "sequence_length": sequence_length,
                "vocab_size": vocab_size,
                "hidden_size": 32,
                "register_tokens": 2,
                "core_layers": 1,
                "low_level_steps": 1,
                "transitions_per_supervision": 1,
                "supervision_steps": 1,
                "core_expansion": 1.0,
                "guidance_expansion": 1.0,
                "decoder_expansion": 1.0,
                "width_multiple": 8,
                "forward_dtype": "float32",
                "attention_heads": 4,
            },
            "candidate_count": 1,
        }
    common = {
        "batch_size": 2,
        "seq_len": sequence_length,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": identifiers,
        "vocab_size": vocab_size,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 16,
        "expansion": 1.0,
        "num_heads": 4,
        "pos_encodings": "none",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }
    if config.model == "fprm":
        common.update(
            n_backwards_L=1,
            norm_type="pre-norm",
            norm_placement="output",
            residual_scale="input-independent",
            alpha_1_init=0.75,
            alpha_2_init=0.25,
            max_iter=1,
            max_iter_eval=1,
            max_iter_dist="det",
            halting_mechanism="fixed_iterations",
            fixed_init=True,
        )
    else:
        common["mlp_t"] = False
    return {"arch": common}


def _metadata_from_source(
    config: TrainConfig,
    stream: PuzzleTrainStream,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if payload is None:
        if config.preset == "sanity":
            return _sanity_model_metadata(config, stream)
        if config.model != "gram":
            raise ValueError("The paper preset requires a source checkpoint")
        return copy.deepcopy(_GRAM_PAPER_METADATA[config.task])
    if payload.get("format") == "RRM_CHECKPOINT_V1":
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, Mapping) or not isinstance(
            resolved.get("model_metadata"), Mapping
        ):
            raise ValueError("RRM checkpoint lacks resolved model metadata")
        return copy.deepcopy(dict(resolved["model_metadata"]))
    if config.model == "gram" and config.preset == "paper":
        return copy.deepcopy(_GRAM_PAPER_METADATA[config.task])
    if config.model == "gram":
        raw = payload.get("config")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
            raise ValueError("GRAM source checkpoint lacks model config")
        values = dict(raw["model"])
        values.setdefault("positionwise_initial_state", False)
        values.setdefault("share_recursive_core", True)
        return {"model": values}
    if config.model == "fprm":
        registered = payload.get("registered_config")
        if isinstance(registered, Mapping):
            pretrain = registered.get("pretrain_config")
            if isinstance(pretrain, Mapping) and isinstance(pretrain.get("arch"), Mapping):
                return {"arch": copy.deepcopy(dict(pretrain["arch"]))}
    if config.model == "ptrm" and isinstance(payload.get("args"), Mapping):
        args = payload["args"]
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("PTRM source checkpoint lacks model state")
        normalized_keys = {
            key.removeprefix("_orig_mod.model.").removeprefix("model."): value
            for key, value in state.items()
        }
        arch = {
            name: args[name]
            for name in (
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
        }
        arch.update(
            batch_size=_runtime_preset(config)["batch_size"],
            seq_len=int(stream.inputs.shape[1]),
            vocab_size=normalized_keys["inner.embed_tokens.embedding_weight"].shape[0],
            num_puzzle_identifiers=normalized_keys[
                "inner.puzzle_emb.weights"
            ].shape[0],
            loss={"q_loss_coeff": float(args.get("q_loss_weight", 0.5))},
        )
        return {"arch": arch}
    if (
        config.model == "ptrm"
        and config.task == "maze-hard"
        and config.fb
        and isinstance(payload.get("model_state_dict"), Mapping)
    ):
        state = payload["model_state_dict"]
        normalized_keys = {
            key.removeprefix("_orig_mod.model.").removeprefix("model."): value
            for key, value in state.items()
        }
        arch = copy.deepcopy(_PTRM_MAZE_PAPER_ARCH)
        arch.update(
            batch_size=_runtime_preset(config)["batch_size"],
            seq_len=int(stream.inputs.shape[1]),
            vocab_size=normalized_keys["inner.embed_tokens.embedding_weight"].shape[0],
            num_puzzle_identifiers=normalized_keys[
                "inner.puzzle_emb.weights"
            ].shape[0],
        )
        return {"arch": arch}
    if config.checkpoint is not None:
        config_path = config.checkpoint.parent / "all_config.yaml"
        if config_path.is_file():
            released = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(released, Mapping) and isinstance(released.get("arch"), Mapping):
                arch = copy.deepcopy(dict(released["arch"]))
                arch.update(
                    batch_size=_runtime_preset(config)["batch_size"],
                    seq_len=int(stream.inputs.shape[1]),
                    num_puzzle_identifiers=int(
                        stream.metadata.get("num_puzzle_identifiers", 1)
                    ),
                    vocab_size=int(stream.metadata["vocab_size"]),
                )
                return {"arch": arch}
    raise ValueError("Unable to resolve model metadata from the source checkpoint")


def _fb_config(config: TrainConfig, runtime: Mapping[str, Any]) -> FenchelBregmanConfig | None:
    if not config.fb:
        return None
    if config.preset == "sanity":
        return FenchelBregmanConfig(
            candidate_count=10,
            temperature=1.0,
            gradient_scale=1.0,
            tracker_beta=0.9,
        )
    if config.model == "fprm" and config.task == "maze-hard":
        return FenchelBregmanConfig.maze_paper()
    if config.model == "ptrm" and config.task == "maze-hard":
        return FenchelBregmanConfig.ptrm_maze_paper()
    if config.model == "ptrm" and config.task == "sudoku-extreme":
        return FenchelBregmanConfig.sudoku_paper(
            global_batch_size=int(runtime["batch_size"])
        )
    raise ValueError("No frozen Fenchel--Bregman preset for this model/task")


def _ema_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def _update_ema(shadow: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            shadow[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
        else:
            shadow[name].copy_(value)


def _gram_state_slice(state: object, start: int, stop: int) -> object:
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("GRAM carry must be a tensor mapping")
    return {
        name: value[start:stop]
        for name, value in state.items()
        if isinstance(value, Tensor)
    }


def _concatenate_gram_states(states: Sequence[object]) -> dict[str, Tensor]:
    if not states or any(not isinstance(state, Mapping) for state in states):
        raise TypeError("GRAM microbatches did not return carry mappings")
    names = tuple(states[0])  # type: ignore[arg-type]
    if any(tuple(state) != names for state in states):  # type: ignore[arg-type]
        raise ValueError("GRAM microbatch carry fields differ")
    return {
        name: torch.cat([state[name] for state in states])  # type: ignore[index]
        for name in names
    }


def _gram_needs_new_batch(carry: object) -> bool:
    if carry is None:
        return True
    if not isinstance(carry, Mapping) or not isinstance(carry.get("halted"), Tensor):
        raise TypeError("GRAM carry lacks halted state")
    return bool(carry["halted"].all().item())


def _state_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {name: _state_to_cpu(item) for name, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_state_to_cpu(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    return copy.deepcopy(value)


def _local_batch(
    batch: Mapping[str, Tensor], context: DistributedContext
) -> dict[str, Tensor]:
    if context.world_size == 1:
        return dict(batch)
    size = next(iter(batch.values())).shape[0]
    if size % context.world_size:
        raise ValueError("global batch size must be divisible by world size")
    local_size = size // context.world_size
    start = context.rank * local_size
    stop = start + local_size
    return {name: value[start:stop] for name, value in batch.items()}


@contextmanager
def _seeded_ivon_parameters(
    optimizer: Optimizer, *, seed: int | None
) -> Iterator[None]:
    if seed is None:
        with optimizer.sampled_params(train=True):  # type: ignore[attr-defined]
            yield
        return
    device = getattr(optimizer, "_device")
    devices = [] if device.type == "cpu" else [device]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        parameter_average, noise = optimizer._sample_params()  # type: ignore[attr-defined]
    try:
        yield
    except BaseException:
        optimizer._restore_param_average(  # type: ignore[attr-defined]
            False, parameter_average, noise
        )
        raise
    else:
        optimizer._restore_param_average(  # type: ignore[attr-defined]
            True, parameter_average, noise
        )


def _optimizer_parameters(optimizer: Optimizer) -> list[nn.Parameter]:
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter is not None
    ]


def _flatten_optimizer_gradients(optimizer: Optimizer) -> Tensor:
    parameters = _optimizer_parameters(optimizer)
    if not parameters:
        raise ValueError("optimizer has no parameters")
    return torch.cat(
        [
            parameter.grad.detach().flatten().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter).flatten()
            for parameter in parameters
        ]
    )


@torch.no_grad()
def _assign_ivon_sample(
    optimizer: Optimizer, parameter_average: Tensor, noise: Tensor
) -> None:
    offset = 0
    for parameter in _optimizer_parameters(optimizer):
        stop = offset + parameter.numel()
        parameter.data = (parameter_average[offset:stop] + noise[offset:stop]).view(
            parameter.shape
        )
        offset = stop
    if offset != parameter_average.numel() or noise.numel() != offset:
        raise ValueError("IVON sample vectors do not match optimizer parameters")


def _capture_torch_rng(device: torch.device) -> tuple[Tensor, Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_torch_rng(
    state: tuple[Tensor, Tensor | None], device: torch.device
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def run_training(config: TrainConfig) -> Path:
    """Run one deterministic training stream and save its exact resume state."""

    if config.max_updates < 1:
        raise ValueError("max_updates must be positive")
    if config.preset == "sanity" and config.max_updates > 1:
        raise ValueError("sanity preset permits at most one update")
    distributed = initialize_distributed(config.device)
    if distributed.primary:
        if config.output.exists() and any(config.output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output: {config.output}"
            )
        config.output.mkdir(parents=True, exist_ok=True)
    if distributed.world_size > 1:
        torch.distributed.barrier()
    seed_all(config.seed)
    runtime = _runtime_preset(config)
    expected_world_size = int(runtime.get("world_size", 1))
    if config.preset == "paper" and distributed.world_size != expected_world_size:
        raise ValueError(
            "paper preset world-size mismatch: "
            f"expected {expected_world_size}, actual {distributed.world_size}"
        )
    runtime["world_size"] = distributed.world_size
    stream = PuzzleTrainStream(
        config.dataset,
        batch_size=int(runtime["batch_size"]),
        seed=config.seed,
        epochs_per_iteration=max(
            1,
            math.ceil(
                config.max_updates
                * int(runtime["batch_size"])
                / int(runtime.get("train_group_count_for_epochs", 1_000))
            ),
        ),
    )
    source_payload: Mapping[str, Any] | None = None
    source_hash: str | None = None
    if config.checkpoint is not None:
        source_hash = sha256_file(config.checkpoint)
        loaded = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping):
            raise ValueError("Source checkpoint must contain a mapping")
        source_payload = loaded
    model_metadata = _metadata_from_source(config, stream, source_payload)
    if config.model in ("fprm", "ptrm") and isinstance(
        model_metadata.get("arch"), Mapping
    ):
        arch = dict(model_metadata["arch"])
        arch["batch_size"] = int(runtime["batch_size"]) // distributed.world_size
        if config.model == "fprm":
            arch["seq_len"] = int(stream.inputs.shape[1])
            arch["vocab_size"] = int(stream.metadata["vocab_size"])
            arch["num_puzzle_identifiers"] = int(
                stream.metadata.get("num_puzzle_identifiers", 1)
            )
        model_metadata["arch"] = arch
    if config.model == "ptrm" and "q_loss_weight" in runtime:
        arch = dict(model_metadata.get("arch", {}))
        loss_metadata = dict(arch.get("loss", {}))
        loss_metadata["q_loss_coeff"] = float(runtime["q_loss_weight"])
        loss_metadata["cellwise_q_loss"] = bool(
            runtime.get("cellwise_q_loss", False)
        )
        loss_metadata["success_loss_aggregation"] = str(
            runtime.get("success_loss_aggregation", "mean")
        )
        loss_metadata["smooth_weakest_delta"] = float(
            runtime.get("smooth_weakest_delta", math.log(2.0))
        )
        arch["loss"] = loss_metadata
        model_metadata["arch"] = arch
    adapter = get_model_adapter(config.model)
    model = adapter.build_model(config.task, config.model, model_metadata)
    device = distributed.device
    model.to(device).train()
    if source_payload is not None and source_payload.get("format") != "RRM_CHECKPOINT_V1":
        if config.model == "gram" and not any(
            key in source_payload
            for key in ("ema_model_state_dict", "ema_state_dict", "config")
        ):
            from rrm.gram import (
                GenerativeRecursiveReasoningModel,
                transfer_trm_weights,
            )

            if not isinstance(model, GenerativeRecursiveReasoningModel):
                raise TypeError("GRAM warm start built the wrong model type")
            source_state = source_payload.get(
                "model_state_dict", source_payload.get("model", source_payload)
            )
            if not isinstance(source_state, Mapping):
                raise ValueError("GRAM warm start lacks a TRM model state")
            transfer_trm_weights(model, source_state)  # type: ignore[arg-type]
        else:
            adapter.load_checkpoint(model, config.checkpoint)  # type: ignore[arg-type]
    if runtime["optimizer"] == "ivon":
        optimizer = build_ivon_optimizer(model, runtime)
        sparse_optimizer = adapter.build_optimizers(model, runtime)[1]
    else:
        optimizer, sparse_optimizer = adapter.build_optimizers(model, runtime)
    if (
        source_payload is not None
        and source_payload.get("format") != "RRM_CHECKPOINT_V1"
        and runtime["optimizer"] == "ivon"
        and isinstance(source_payload.get("optimizer"), Mapping)
        and isinstance(source_payload.get("ivon_current_step"), int)
    ):
        load_ivon_optimizer_state(
            optimizer,
            source_payload["optimizer"],
            current_step=int(source_payload["ivon_current_step"]),
        )
        if sparse_optimizer is not None:
            sparse_state = source_payload.get("sparse_optimizer")
            if not isinstance(sparse_state, Mapping):
                raise ValueError("FPRM initializer lacks sparse optimizer state")
            sparse_optimizer.load_state_dict(sparse_state)
    fb_config = _fb_config(config, runtime)
    tracker = (
        FailureProbabilityTracker(
            stream.num_groups,
            beta=fb_config.tracker_beta,
            initial_failure=fb_config.initial_failure,
            device=device,
            dtype=(
                torch.float64
                if fb_config.duplicate_update == "group_mean"
                else torch.float32
            ),
        )
        if fb_config is not None
        else None
    )
    start_step = 0
    carry: object = None
    ema = _ema_state(model) if "ema_decay" in runtime else None
    if source_payload is not None and source_payload.get("format") == "RRM_CHECKPOINT_V1":
        start_step, adapter_state = restore_training_checkpoint(
            source_payload,
            config=config,
            model=model,
            optimizer=optimizer,
            tracker=tracker,
        )
        carry = adapter_state.get("carry")
        stream_state = adapter_state.get("stream")
        if not isinstance(stream_state, Mapping):
            raise ValueError("Resume checkpoint is missing stream state")
        stream.load_state_dict(stream_state)
        if sparse_optimizer is not None:
            sparse_state = adapter_state.get("sparse_optimizer")
            if not isinstance(sparse_state, Mapping):
                raise ValueError("Resume checkpoint is missing sparse optimizer state")
            sparse_optimizer.load_state_dict(sparse_state)
        if ema is not None:
            raw_ema = adapter_state.get("ema_model_state")
            if not isinstance(raw_ema, Mapping):
                raise ValueError("Resume checkpoint is missing EMA state")
            ema = {name: value.to(device) for name, value in raw_ema.items()}
    resolved_config = {
        "runtime": runtime,
        "model_metadata": model_metadata,
        "fb_config": None if fb_config is None else asdict(fb_config),
    }
    last_metrics: dict[str, Tensor] = {}
    for step in range(start_step + 1, config.max_updates + 1):
        if config.model != "gram" or _gram_needs_new_batch(carry):
            cpu_batch = _local_batch(stream.next_batch(), distributed)
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
        else:
            assert isinstance(carry, Mapping)
            batch = {
                "inputs": carry["inputs"],
                "labels": carry["labels"],
                "puzzle_identifiers": torch.zeros(
                    int(carry["inputs"].shape[0]), dtype=torch.long, device=device
                ),
                "group_ids": torch.zeros(
                    int(carry["inputs"].shape[0]), dtype=torch.long, device=device
                ),
            }
        group_ids = batch["group_ids"]
        native_batch = batch
        if config.model == "gram":
            optimizer.zero_grad(set_to_none=True)
            micro_batch_size = int(runtime.get("micro_batch_size", runtime["batch_size"]))
            state_parts: list[object] = []
            metric_sums: dict[str, Tensor] = {}
            for start in range(0, int(runtime["batch_size"]), micro_batch_size):
                stop = start + micro_batch_size
                micro_batch = {
                    name: value[start:stop] for name, value in native_batch.items()
                }
                output = adapter.training_forward(
                    model, _gram_state_slice(carry, start, stop), micro_batch
                )
                loss, micro_metrics = compute_training_loss(
                    output,
                    fb_config=fb_config,
                    tracker=tracker,
                    group_ids=micro_batch["group_ids"],
                )
                (loss / int(runtime["batch_size"])).backward()
                state_parts.append(output.state)
                for name, value in micro_metrics.items():
                    detached = value.detach()
                    metric_sums[name] = metric_sums.get(
                        name, torch.zeros_like(detached)
                    ) + detached
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(runtime.get("gradient_clip", float("inf")))
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("GRAM produced a non-finite gradient norm")
            warmup = int(runtime.get("learning_rate_warmup_steps", 0))
            learning_rate = float(runtime["learning_rate"])
            if warmup:
                learning_rate *= min(step / warmup, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            carry = _concatenate_gram_states(state_parts)
            last_metrics = metric_sums
            if ema is not None:
                _update_ema(ema, model, float(runtime["ema_decay"]))
            continue
        if carry is None and hasattr(model, "initial_carry"):
            carry = model.initial_carry(native_batch)  # type: ignore[attr-defined]
        if runtime["optimizer"] == "ivon":
            if int(runtime.get("ivon_mc_samples", 1)) != 1:
                raise ValueError("The public FB runtime requires IVON MC=1")
            carry_start = copy.deepcopy(carry)
            closure_output: TrainingOutput | None = None
            closure_metrics: dict[str, Tensor] = {}

            def closure() -> Tensor:
                nonlocal closure_output, closure_metrics
                optimizer.zero_grad()
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad()
                sample_seed = (
                    1_000_003 + step
                    if config.model == "fprm" and config.task == "maze-hard"
                    else None
                )
                with _seeded_ivon_parameters(optimizer, seed=sample_seed):
                    closure_output = adapter.training_forward(
                        model, copy.deepcopy(carry_start), native_batch
                    )
                    effective_ids = getattr(
                        closure_output.state, "current_data", {}
                    ).get("group_ids", group_ids)
                    loss, closure_metrics = compute_training_loss(
                        closure_output,
                        fb_config=fb_config,
                        tracker=tracker,
                        group_ids=effective_ids,
                        update_tracker=distributed.world_size == 1,
                    )
                    if tracker is not None and distributed.world_size > 1:
                        gathered_ids, gathered_failures = gather_rank_ordered_vectors(
                            effective_ids,
                            closure_metrics["fb_failure"],
                            context=distributed,
                        )
                        tracker.update(
                            gathered_ids,
                            gathered_failures,
                            duplicate_update=fb_config.duplicate_update,
                        )
                    normalized_loss = loss / int(runtime["batch_size"])
                    normalized_loss.backward()
                    sum_distributed_gradients(model, distributed)
                return normalized_loss

            if fb_config is not None and fb_config.price_proxy_scale is not None:
                optimizer.zero_grad()
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad()
                parameter_average, noise = optimizer._sample_params()  # type: ignore[attr-defined]
                sampled_rng = _capture_torch_rng(device)
                ordinary_gradient: Tensor | None = None
                mean_output: TrainingOutput | None = None
                mean_metrics: dict[str, Tensor] = {}
                try:
                    if should_refresh_curvature(
                        step, fb_config.curvature_refresh_interval
                    ):
                        ordinary_output = adapter.training_forward(
                            model, copy.deepcopy(carry_start), native_batch
                        )
                        ordinary_loss, _ = compute_training_loss(
                            ordinary_output,
                            fb_config=None,
                            tracker=None,
                            group_ids=group_ids,
                        )
                        (ordinary_loss / int(runtime["batch_size"])).backward()
                        ordinary_gradient = _flatten_optimizer_gradients(optimizer)
                        optimizer.zero_grad()
                        if sparse_optimizer is not None:
                            sparse_optimizer.zero_grad()
                        _assign_ivon_sample(optimizer, parameter_average, noise)
                        _restore_torch_rng(sampled_rng, device)
                    mean_output = adapter.training_forward(
                        model, copy.deepcopy(carry_start), native_batch
                    )
                    mean_ids = getattr(mean_output.state, "current_data", {}).get(
                        "group_ids", group_ids
                    )
                    mean_loss, mean_metrics = compute_training_loss(
                        mean_output,
                        fb_config=fb_config,
                        tracker=tracker,
                        group_ids=mean_ids,
                        update_tracker=False,
                    )
                    normalized_mean_loss = mean_loss / int(runtime["batch_size"])
                    normalized_mean_loss.backward()
                    mean_gradient = _flatten_optimizer_gradients(optimizer)
                finally:
                    optimizer._restore_param_average(  # type: ignore[attr-defined]
                        False, parameter_average, noise
                    )
                curvature_gradient, _ = price_anchored_curvature_gradient(
                    mean_gradient,
                    ordinary_gradient=ordinary_gradient,
                    step=step,
                    config=fb_config,
                )
                optimizer.step_from_split_gradient(  # type: ignore[attr-defined]
                    mean_gradient,
                    curvature_gradient,
                    noise,
                    objective=normalized_mean_loss,
                )
                assert mean_output is not None and tracker is not None
                tracker.update(
                    mean_ids,
                    mean_metrics["fb_failure"],
                    duplicate_update=fb_config.duplicate_update,
                )
                carry = mean_output.state
                last_metrics = mean_metrics
            else:
                optimizer.step(closure)
                assert closure_output is not None
                carry = closure_output.state
                last_metrics = closure_metrics
        else:
            optimizer.zero_grad()
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad()
            output = adapter.training_forward(model, carry, native_batch)
            effective_ids = getattr(output.state, "current_data", {}).get(
                "group_ids", group_ids
            )
            loss, last_metrics = compute_training_loss(
                output,
                fb_config=fb_config,
                tracker=tracker,
                group_ids=effective_ids,
            )
            (loss / int(runtime["batch_size"])).backward()
            optimizer.step()
            carry = output.state
        if sparse_optimizer is not None:
            sparse_optimizer.step()
        if ema is not None:
            _update_ema(ema, model, float(runtime["ema_decay"]))
    adapter_state = {
        "carry": carry,
        "stream": stream.state_dict(),
        "sparse_optimizer": (
            None if sparse_optimizer is None else sparse_optimizer.state_dict()
        ),
        "ema_model_state": ema,
    }
    checkpoint_path = config.output / "checkpoint.pt"
    checkpoint_adapter_state: Mapping[str, Any] = adapter_state
    checkpoint_rng_state: Mapping[str, Any] | None = None
    if distributed.world_size > 1:
        local_bundle = (
            _state_to_cpu(adapter_state),
            _state_to_cpu(capture_rng_state()),
        )
        gathered: list[Any] = [None] * distributed.world_size
        torch.distributed.all_gather_object(gathered, local_bundle)
        checkpoint_adapter_state = {
            "format": "RRM_DISTRIBUTED_STATE_V1",
            "world_size": distributed.world_size,
            "per_rank": [item[0] for item in gathered],
        }
        checkpoint_rng_state = {
            "format": "RRM_DISTRIBUTED_STATE_V1",
            "world_size": distributed.world_size,
            "per_rank": [item[1] for item in gathered],
        }
    if distributed.primary:
        checkpoint_payload = build_checkpoint(
            config=config,
            step=config.max_updates,
            model=model,
            optimizer=optimizer,
            tracker=tracker,
            resolved_config=resolved_config,
            source_checkpoint_sha256=source_hash,
            adapter_state=checkpoint_adapter_state,
        )
        if checkpoint_rng_state is not None:
            checkpoint_payload["rng_state"] = checkpoint_rng_state
        checkpoint_payload["last_metrics"] = {
            name: value.detach().cpu() for name, value in last_metrics.items()
        }
        atomic_torch_save(checkpoint_path, checkpoint_payload)
        atomic_write_json(
            config.output / "resolved_config.json",
            {
                "model": config.model,
                "task": config.task,
                "preset": config.preset,
                "fb": config.fb,
                "seed": config.seed,
                "max_updates": config.max_updates,
                "source_checkpoint_sha256": source_hash,
                "runtime": runtime,
                "fb_config": None if fb_config is None else asdict(fb_config),
            },
        )
    if distributed.world_size > 1:
        torch.distributed.barrier()
    return checkpoint_path


def main(argv: Sequence[str] | None = None) -> None:
    config = config_from_args(build_parser().parse_args(argv))
    checkpoint = run_training(config)
    print(checkpoint)


if __name__ == "__main__":
    main()
