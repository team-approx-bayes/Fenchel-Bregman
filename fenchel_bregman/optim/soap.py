"""SOAP optimizer with a checkpoint-safe, mixed-precision implementation.

This module is derived from the official SOAP implementation at
https://github.com/nikhilvyas/SOAP, commit
``a1e553530fde97d0e6b307d7c82ac6d38b072340``.

Copyright (c) 2024 Nikhil Vyas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The implementation keeps the optimizer state in the Shampoo eigenbasis while
adding argument validation, closure support under ``torch.no_grad``, and
explicit state dtypes.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import chain
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class SOAP(Optimizer):
    """ShampoO with Adam in the preconditioner's eigenbasis."""

    optimizer_name = "soap"

    def __init__(
        self,
        params: Any,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        cast_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        if not (shampoo_beta == -1 or 0 <= shampoo_beta < 1):
            raise ValueError(f"Invalid shampoo_beta: {shampoo_beta}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if precondition_frequency <= 0:
            raise ValueError("precondition_frequency must be positive")
        if max_precond_dim <= 0:
            raise ValueError("max_precond_dim must be positive")
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"Unsupported data format: {data_format}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.cast_dtype = cast_dtype
        self._data_format = data_format

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state while preserving the configured optimizer work dtype.

        ``torch.optim.Optimizer`` normally casts floating state to each parameter's
        dtype. SOAP intentionally keeps its moments and Shampoo matrices in
        ``cast_dtype``, so mixed-precision resumes need an explicit recast.
        """
        super().load_state_dict(state_dict)
        if self.cast_dtype is None:
            return

        def recast(value: Any) -> Any:
            if isinstance(value, Tensor) and value.is_floating_point():
                return value.to(dtype=self.cast_dtype)
            if isinstance(value, list):
                return [recast(item) for item in value]
            if isinstance(value, tuple):
                return tuple(recast(item) for item in value)
            if isinstance(value, dict):
                return {key: recast(item) for key, item in value.items()}
            return value

        for parameter, state in list(self.state.items()):
            if isinstance(parameter, Tensor):
                self.state[parameter] = recast(state)

    @contextmanager
    def sampled_params(self, train: bool = True):
        """Provide the posterior-optimizer interface without perturbing weights."""
        del train
        yield

    @torch.no_grad()
    def step(self, closure=None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                work_dtype = self.cast_dtype or grad.dtype
                grad_work = grad.to(work_dtype)
                state = self.state[parameter]
                state.setdefault("step", 0)
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad_work)
                    state["exp_avg_sq"] = torch.zeros_like(grad_work)

                if "Q" not in state:
                    self._init_preconditioner(
                        grad_work,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else beta2
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self._update_preconditioner(
                        grad_work,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    # SOAP initializes its basis from the first gradient and
                    # deliberately performs no parameter update on that step.
                    continue

                grad_projected = self._project(
                    grad_work,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_projected, grad_projected, value=1 - beta2
                )

                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    correction1 = 1 - beta1 ** state["step"]
                    correction2 = 1 - beta2 ** state["step"]
                    step_size *= correction2**0.5 / correction1

                update = self._project_back(
                    exp_avg / denominator,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )
                if group["normalize_grads"]:
                    update = update / update.square().mean().sqrt().clamp_min(group["eps"])

                # AdamW-style decoupled weight decay.
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-step_size)

                self._update_preconditioner(
                    grad_work,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )
        return loss

    def _merge_dims(self, tensor: Tensor, max_precond_dim: int) -> Tensor:
        if self._data_format == "channels_last" and tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        new_shape: list[int] = []
        current = 1
        for size in tensor.shape:
            candidate = current * size
            if candidate > max_precond_dim:
                if current > 1:
                    new_shape.append(current)
                    current = size
                else:
                    new_shape.append(size)
                    current = 1
            else:
                current = candidate
        if current > 1 or not new_shape:
            new_shape.append(current)
        return tensor.reshape(new_shape)

    def _init_preconditioner(
        self,
        grad: Tensor,
        state: dict[str, Any],
        *,
        precondition_frequency: int,
        shampoo_beta: float,
        max_precond_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
    ) -> None:
        state["GG"] = []
        if grad.ndim == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"].append(
                    torch.zeros(
                        grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype
                    )
                )
            else:
                state["GG"].append([])
        else:
            shaped_grad = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for size in shaped_grad.shape:
                if size <= max_precond_dim:
                    state["GG"].append(
                        torch.zeros(size, size, device=grad.device, dtype=grad.dtype)
                    )
                else:
                    state["GG"].append([])
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _update_preconditioner(
        self,
        grad: Tensor,
        state: dict[str, Any],
        *,
        max_precond_dim: int,
        merge_dims: bool,
        precondition_1d: bool,
    ) -> None:
        due = (
            state["Q"] is not None
            and state["step"] > 0
            and state["step"] % state["precondition_frequency"] == 0
        )
        if due:
            state["exp_avg"] = self._project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

        if grad.ndim == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0),
                    1 - state["shampoo_beta"],
                )
        else:
            shaped_grad = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for axis, size in enumerate(shaped_grad.shape):
                if size > max_precond_dim:
                    continue
                contracted_axes = [
                    *chain(range(axis), range(axis + 1, shaped_grad.ndim))
                ]
                outer = torch.tensordot(
                    shaped_grad,
                    shaped_grad,
                    dims=[contracted_axes, contracted_axes],
                )
                state["GG"][axis].lerp_(outer, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = _get_orthogonal_matrix(state["GG"])
        elif due:
            second_moment = state["exp_avg_sq"]
            original_shape = second_moment.shape
            if merge_dims:
                if self._data_format == "channels_last" and second_moment.ndim == 4:
                    permuted_shape = second_moment.permute(0, 3, 1, 2).shape
                second_moment = self._merge_dims(second_moment, max_precond_dim)
            state["Q"], second_moment = _get_orthogonal_matrix_qr(
                state["GG"], state["Q"], second_moment
            )
            if merge_dims:
                if self._data_format == "channels_last" and len(original_shape) == 4:
                    second_moment = second_moment.reshape(permuted_shape).permute(
                        0, 2, 3, 1
                    )
                else:
                    second_moment = second_moment.reshape(original_shape)
            state["exp_avg_sq"] = second_moment
            state["exp_avg"] = self._project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def _project(
        self,
        tensor: Tensor,
        state: dict[str, Any],
        *,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> Tensor:
        original_shape = tensor.shape
        if merge_dims:
            if self._data_format == "channels_last" and tensor.ndim == 4:
                permuted_shape = tensor.permute(0, 3, 1, 2).shape
            tensor = self._merge_dims(tensor, max_precond_dim)
        for basis in state["Q"]:
            if isinstance(basis, Tensor) and basis.numel():
                tensor = torch.tensordot(tensor, basis.to(tensor.dtype), dims=[[0], [0]])
            else:
                tensor = tensor.permute([*range(1, tensor.ndim), 0])
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                tensor = tensor.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                tensor = tensor.reshape(original_shape)
        return tensor

    def _project_back(
        self,
        tensor: Tensor,
        state: dict[str, Any],
        *,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> Tensor:
        original_shape = tensor.shape
        if merge_dims:
            if self._data_format == "channels_last" and tensor.ndim == 4:
                permuted_shape = tensor.permute(0, 3, 1, 2).shape
            tensor = self._merge_dims(tensor, max_precond_dim)
        for basis in state["Q"]:
            if isinstance(basis, Tensor) and basis.numel():
                tensor = torch.tensordot(tensor, basis.to(tensor.dtype), dims=[[0], [1]])
            else:
                tensor = tensor.permute([*range(1, tensor.ndim), 0])
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                tensor = tensor.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                tensor = tensor.reshape(original_shape)
        return tensor


def _get_orthogonal_matrix(statistics: list[Any]) -> list[Any]:
    bases: list[Any] = []
    for statistic in statistics:
        if not isinstance(statistic, Tensor) or not statistic.numel():
            bases.append([])
            continue
        matrix = statistic.float()
        identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        try:
            _, basis = torch.linalg.eigh(matrix + 1e-30 * identity)
        except RuntimeError:
            _, basis = torch.linalg.eigh(
                matrix.double() + 1e-30 * identity.double()
            )
            basis = basis.float()
        bases.append(torch.flip(basis, dims=[1]))
    return bases


def _get_orthogonal_matrix_qr(
    statistics: list[Any],
    old_bases: list[Any],
    second_moment: Tensor,
) -> tuple[list[Any], Tensor]:
    new_bases: list[Any] = []
    reordered_moment = second_moment
    for axis, (statistic, old_basis) in enumerate(zip(statistics, old_bases)):
        if not isinstance(statistic, Tensor) or not statistic.numel():
            new_bases.append([])
            continue
        matrix = statistic.float()
        basis = old_basis.float()
        estimated_eigenvalues = torch.diag(basis.T @ matrix @ basis)
        order = torch.argsort(estimated_eigenvalues, descending=True)
        reordered_moment = reordered_moment.index_select(axis, order)
        basis = basis[:, order]
        new_basis, _ = torch.linalg.qr(matrix @ basis)
        new_bases.append(new_basis.to(old_basis.dtype))
    return new_bases, reordered_moment
