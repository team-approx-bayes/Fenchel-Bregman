"""Evaluation controls for EVON's eigenspace Gaussian posterior."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.optim import Optimizer


def _project_variance_back(
    variance: Tensor,
    bases: list[Tensor | list[Any]],
) -> Tensor:
    """Map diagonal eigenspace variance to original-coordinate marginals."""
    result = variance
    for basis in bases:
        if len(basis) > 0:
            squared = basis.square().to(device=result.device, dtype=result.dtype)
            result = torch.tensordot(result, squared, dims=[[0], [1]])
        else:
            result = result.permute(list(range(1, result.ndim)) + [0])
    return result


def evon_original_marginal_variance(
    optimizer: Optimizer,
    group: dict[str, Any],
    parameter: Tensor,
) -> Tensor:
    """Return diag(Q diag(v) Q^T) in the parameter's original coordinates."""
    state = optimizer.state[parameter]
    work_dtype = getattr(optimizer, "cast_dtype", None) or parameter.dtype
    hess_init = float(getattr(optimizer, "_hess_init"))
    hessian = state.get(
        "h_mom",
        torch.full_like(parameter, hess_init, dtype=work_dtype),
    )
    variance = (float(group["ess"]) * (hessian + float(group["weight_decay"]))).reciprocal()
    bases = state.get("Q")
    if bases is None:
        return variance.reshape(parameter.shape)

    original_shape = variance.shape
    if group.get("merge_dims", False):
        variance = optimizer._merge_dims(variance, int(group["max_precond_dim"]))
    variance = _project_variance_back(variance, bases)
    if group.get("merge_dims", False):
        variance = variance.reshape(original_shape)
    return variance.reshape(parameter.shape)


def prepare_evon_same_marginal_stds(optimizer: Optimizer) -> list[Tensor]:
    """Precompute independent-coordinate stds with EVON's exact marginals."""
    if getattr(optimizer, "optimizer_name", None) != "evon":
        raise ValueError("EVON same-marginal control requires an EVON optimizer.")
    controls: list[Tensor] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter is None or not parameter.requires_grad:
                continue
            controls.append(evon_original_marginal_variance(optimizer, group, parameter).sqrt())
    return controls


@contextmanager
def evon_same_marginal_sampling_context(
    optimizer: Optimizer,
    *,
    standard_deviations: list[Tensor],
    posterior_scale: float,
) -> Iterator[None]:
    """Sample an original-coordinate diagonal Gaussian around the EVON mean."""
    if getattr(optimizer, "optimizer_name", None) != "evon":
        raise ValueError("EVON same-marginal control requires an EVON optimizer.")
    if posterior_scale < 0:
        raise ValueError("posterior scale must be non-negative")
    if posterior_scale == 0:
        yield
        return

    saved_parameters: list[tuple[Tensor, Tensor]] = []
    control_index = 0
    try:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter is None or not parameter.requires_grad:
                    continue
                saved_parameters.append((parameter, parameter.detach().clone()))
                std = standard_deviations[control_index]
                noise = torch.randn(std.shape, device=std.device, dtype=std.dtype).mul_(std)
                parameter.data.add_(noise.to(parameter.dtype), alpha=posterior_scale)
                control_index += 1
        if control_index != len(standard_deviations):
            raise ValueError("EVON same-marginal control count does not match optimizer parameters.")
        yield
    finally:
        for parameter, saved in saved_parameters:
            parameter.data.copy_(saved)
