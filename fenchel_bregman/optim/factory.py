"""Optimizer construction and posterior-sampling capabilities."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from optim.ivon import IVON
from optim.soap import SOAP


SUPPORTED_OPTIMIZERS = ("ivon", "soap", "evon", "adamatan2")


def add_optimizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, default="ivon")

    parser.add_argument("--soap-beta2", type=float, default=0.95)
    parser.add_argument("--soap-shampoo-beta", type=float, default=-1.0)
    parser.add_argument("--soap-eps", type=float, default=1e-8)
    parser.add_argument("--soap-weight-decay", type=float, default=0.01)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precond-dim", type=int, default=10000)
    parser.add_argument("--soap-merge-dims", action="store_true")
    parser.add_argument("--soap-precondition-1d", action="store_true")
    parser.add_argument("--soap-normalize-grads", action="store_true")
    parser.add_argument("--soap-no-correct-bias", action="store_true")

    parser.add_argument("--evon-ess", type=float, default=None)
    parser.add_argument("--evon-hess-init", type=float, default=None)
    parser.add_argument("--evon-beta2", type=float, default=0.9999)
    parser.add_argument("--evon-weight-decay", type=float, default=None)
    parser.add_argument("--evon-shampoo-beta", type=float, default=-1.0)
    parser.add_argument("--evon-eps", type=float, default=1e-10)
    parser.add_argument("--evon-precondition-frequency", type=int, default=10)
    parser.add_argument("--evon-max-precond-dim", type=int, default=10000)
    parser.add_argument("--evon-merge-dims", action="store_true")
    parser.add_argument("--evon-precondition-1d", action="store_true")
    parser.add_argument("--evon-phasing", action="store_true")
    parser.add_argument("--evon-price-clip-ratio", type=float, default=None)
    parser.add_argument("--evon-no-whiten-prec-grad", action="store_true")
    parser.add_argument("--evon-sync", action="store_true")
    parser.add_argument("--evon-debias-beta2", action="store_true")
    parser.add_argument("--evon-no-correct-bias", action="store_true")
    parser.add_argument(
        "--evon-cast-dtype",
        choices=("float32", "bfloat16", "parameter"),
        default="float32",
    )


def optimizer_name(optimizer: Optimizer) -> str:
    explicit = getattr(optimizer, "optimizer_name", None)
    if isinstance(explicit, str):
        return explicit
    if isinstance(optimizer, IVON):
        return "ivon"
    return optimizer.__class__.__name__.lower()


def _value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _cast_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value in (None, "parameter"):
        return None
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value))


def create_optimizer(
    name: str,
    named_parameters: Sequence[tuple[str, Tensor] | Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Optimizer:
    name = name.lower()
    if name == "adam_atan2":
        name = "adamatan2"
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name}")
    if not named_parameters:
        raise ValueError("Optimizer requires at least one parameter")
    if isinstance(named_parameters[0], Mapping):
        parameter_groups = []
        for source_group in named_parameters:
            if not isinstance(source_group, Mapping):
                raise TypeError("Optimizer parameter groups cannot mix input formats")
            parameters = list(source_group.get("params", ()))
            parameter_names = list(source_group.get("param_names", ()))
            if len(parameters) != len(parameter_names):
                raise ValueError("Each optimizer parameter requires a param_name")
            parameter_groups.append(
                {"params": parameters, "param_names": parameter_names}
            )
    else:
        if any(isinstance(item, Mapping) for item in named_parameters):
            raise TypeError("Optimizer parameter groups cannot mix input formats")
        flat_named_parameters = list(named_parameters)
        parameter_groups = [
            {
                "params": [parameter for _, parameter in flat_named_parameters],
                "param_names": [
                    parameter_name for parameter_name, _ in flat_named_parameters
                ],
            }
        ]

    if name == "ivon":
        return IVON(
            parameter_groups,
            lr=float(_value(config, "lr", 1e-4)),
            ess=float(_value(config, "ivon_ess", 1e5)),
            hess_init=float(_value(config, "ivon_hess_init", 1.0)),
            beta1=float(_value(config, "beta1", 0.9)),
            beta2=float(_value(config, "ivon_beta2", 0.99999)),
            weight_decay=float(_value(config, "ivon_weight_decay", 1e-4)),
            mc_samples=int(_value(config, "ivon_mc_samples", 1)),
            hess_approx=str(_value(config, "ivon_hess_approx", "price")),
            clip_radius=float(_value(config, "ivon_clip_radius", float("inf"))),
            debias=not bool(_value(config, "no_ivon_debias", False)),
            rescale_lr=not bool(_value(config, "no_ivon_rescale_lr", False)),
            update_transform=str(_value(config, "ivon_update_transform", "clip")),
            muon_whiten_eps=float(_value(config, "ivon_muon_whiten_eps", 1e-8)),
            muon_ns_steps=int(_value(config, "ivon_muon_ns_steps", 5)),
            pdr_curvature_lambda=float(_value(config, "pdr_curvature_lambda", 0.0)),
            pdr_target_k=int(_value(config, "pdr_target_k", 10)),
            pdr_curvature_eps=float(_value(config, "pdr_curvature_eps", 1e-12)),
            pdr_curvature_ratio=float(_value(config, "pdr_curvature_ratio", 0.0)),
        )

    if name == "soap":
        return SOAP(
            parameter_groups,
            lr=float(_value(config, "lr", 3e-3)),
            betas=(
                float(_value(config, "beta1", 0.95)),
                float(_value(config, "soap_beta2", 0.95)),
            ),
            shampoo_beta=float(_value(config, "soap_shampoo_beta", -1.0)),
            eps=float(_value(config, "soap_eps", 1e-8)),
            weight_decay=float(_value(config, "soap_weight_decay", 0.01)),
            precondition_frequency=int(
                _value(config, "soap_precondition_frequency", 10)
            ),
            max_precond_dim=int(_value(config, "soap_max_precond_dim", 10000)),
            merge_dims=bool(_value(config, "soap_merge_dims", False)),
            precondition_1d=bool(_value(config, "soap_precondition_1d", False)),
            normalize_grads=bool(_value(config, "soap_normalize_grads", False)),
            correct_bias=not bool(_value(config, "soap_no_correct_bias", False)),
            cast_dtype=torch.float32,
        )

    if name == "adamatan2":
        try:
            from adam_atan2 import AdamATan2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AdamATan2 requires the 'adam_atan2' package in the active environment."
            ) from exc
        return AdamATan2(
            parameter_groups,
            lr=float(_value(config, "lr", 1e-4)),
            betas=(
                float(_value(config, "beta1", 0.9)),
                float(_value(config, "beta2", 0.999)),
            ),
            weight_decay=float(_value(config, "weight_decay", 0.0)),
        )
    try:
        from optim.evon import EVON
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EVON requires the official 'evon' package. Install requirements.txt."
        ) from exc

    return EVON(
        parameter_groups,
        lr=float(_value(config, "lr", 3e-3)),
        ess=float(_value(config, "evon_ess", _value(config, "ivon_ess", 1e5))),
        hess_init=float(
            _value(config, "evon_hess_init", _value(config, "ivon_hess_init", 1.0))
        ),
        betas=(
            float(_value(config, "beta1", 0.95)),
            float(_value(config, "evon_beta2", 0.9999)),
        ),
        shampoo_beta=float(_value(config, "evon_shampoo_beta", -1.0)),
        eps=float(_value(config, "evon_eps", 1e-10)),
        weight_decay=float(
            _value(config, "evon_weight_decay", _value(config, "ivon_weight_decay", 1e-6))
        ),
        precondition_frequency=int(
            _value(config, "evon_precondition_frequency", 10)
        ),
        max_precond_dim=int(_value(config, "evon_max_precond_dim", 10000)),
        merge_dims=bool(_value(config, "evon_merge_dims", False)),
        precondition_1d=bool(_value(config, "evon_precondition_1d", False)),
        correct_bias=not bool(_value(config, "evon_no_correct_bias", False)),
        cast_dtype=_cast_dtype(_value(config, "evon_cast_dtype", "float32")),
        mc_samples=int(_value(config, "ivon_mc_samples", 1)),
        phasing=bool(_value(config, "evon_phasing", False)),
        price_clip_ratio=_value(config, "evon_price_clip_ratio", None),
        sync=bool(_value(config, "evon_sync", False)),
        whiten_prec_grad=not bool(
            _value(config, "evon_no_whiten_prec_grad", False)
        ),
        debias_beta2=bool(_value(config, "evon_debias_beta2", False)),
    )


def create_dense_optimizer(
    named_parameters: Sequence[tuple[str, Tensor]], args: argparse.Namespace
) -> Optimizer:
    return create_optimizer(args.optimizer, named_parameters, vars(args))


def supports_posterior_sampling(optimizer: Optimizer) -> bool:
    return optimizer_name(optimizer) in ("ivon", "evon")


@contextmanager
def optimizer_sampling_context(
    optimizer: Optimizer,
    *,
    train: bool,
    posterior_scale: float = 1.0,
):
    name = optimizer_name(optimizer)
    if name in ("soap", "adamatan2"):
        if posterior_scale not in (0.0, 1.0):
            raise ValueError(f"{optimizer.__class__.__name__} has no posterior scale")
        yield
        return
    if posterior_scale < 0:
        raise ValueError("posterior scale must be non-negative")
    if posterior_scale == 0:
        yield
        return
    if name == "evon" and hasattr(optimizer, "sampled_params_scaled"):
        with optimizer.sampled_params_scaled(train=train, scale=posterior_scale):
            yield
        return

    original_ess = [group["ess"] for group in optimizer.param_groups]
    try:
        if posterior_scale != 1:
            for group, ess in zip(optimizer.param_groups, original_ess):
                group["ess"] = ess / (posterior_scale * posterior_scale)
        with optimizer.sampled_params(train=train):
            yield
    finally:
        for group, ess in zip(optimizer.param_groups, original_ess):
            group["ess"] = ess


def move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    def move(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    for key, value in list(optimizer.state.items()):
        optimizer.state[key] = move(value)
    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if key not in ("params", "param_names"):
                group[key] = move(value)
