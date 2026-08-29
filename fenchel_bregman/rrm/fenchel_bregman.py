"""Model-independent Fenchel--Bregman objective and failure state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor


def half_life_to_beta(half_life: int) -> float:
    """Convert a visit half-life to the registered EMA coefficient."""

    if isinstance(half_life, bool) or int(half_life) != half_life or half_life < 1:
        raise ValueError("half_life must be a positive integer")
    return 2.0 ** (-1.0 / int(half_life))


@dataclass(frozen=True)
class FenchelBregmanConfig:
    candidate_count: int
    temperature: float
    gradient_scale: float
    tracker_beta: float
    initial_failure: float = 0.5
    duplicate_update: str = "sequential"
    curvature_refresh_interval: int | None = None
    price_proxy_scale: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_count, bool)
            or int(self.candidate_count) != self.candidate_count
            or self.candidate_count < 2
        ):
            raise ValueError("candidate_count must be an integer of at least two")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(self.gradient_scale) or self.gradient_scale <= 0:
            raise ValueError("gradient_scale must be positive and finite")
        if not math.isfinite(self.tracker_beta) or not 0 <= self.tracker_beta < 1:
            raise ValueError("tracker_beta must lie in [0, 1)")
        if not math.isfinite(self.initial_failure) or not 0 <= self.initial_failure <= 1:
            raise ValueError("initial_failure must lie in [0, 1]")
        if self.duplicate_update not in ("sequential", "group_mean"):
            raise ValueError("duplicate_update must be sequential or group_mean")
        if self.curvature_refresh_interval is not None and (
            isinstance(self.curvature_refresh_interval, bool)
            or int(self.curvature_refresh_interval) != self.curvature_refresh_interval
            or self.curvature_refresh_interval < 1
        ):
            raise ValueError("curvature_refresh_interval must be positive")
        if self.price_proxy_scale is not None and (
            not math.isfinite(self.price_proxy_scale) or self.price_proxy_scale <= 0
        ):
            raise ValueError("price_proxy_scale must be positive and finite")
        if (self.curvature_refresh_interval is None) != (self.price_proxy_scale is None):
            raise ValueError(
                "price proxy scale and curvature refresh interval must be set together"
            )

    @property
    def curvature_mode(self) -> str:
        return (
            "price_proxy_with_ordinary_refresh"
            if self.price_proxy_scale is not None
            else "ordinary"
        )

    @classmethod
    def maze_paper(cls) -> "FenchelBregmanConfig":
        return cls(
            candidate_count=10,
            temperature=0.0625,
            gradient_scale=507.9159789337636,
            tracker_beta=half_life_to_beta(16),
            initial_failure=0.5,
            duplicate_update="sequential",
        )

    @classmethod
    def sudoku_paper(
        cls, *, global_batch_size: int = 128
    ) -> "FenchelBregmanConfig":
        if (
            isinstance(global_batch_size, bool)
            or int(global_batch_size) != global_batch_size
            or global_batch_size < 1
        ):
            raise ValueError("global_batch_size must be a positive integer")
        return cls(
            candidate_count=10,
            temperature=1.0,
            # The shared training loop applies the registered global-batch
            # normalization after adding the untouched Q objective.
            gradient_scale=1.0,
            tracker_beta=0.99,
            initial_failure=0.5,
            duplicate_update="group_mean",
            curvature_refresh_interval=20,
            price_proxy_scale=1.0842683683055117,
        )

    @classmethod
    def ptrm_maze_paper(cls) -> "FenchelBregmanConfig":
        """Return the frozen PTRM Maze-Hard finite-K configuration."""

        return cls(
            candidate_count=10,
            temperature=1.0,
            gradient_scale=1.0,
            tracker_beta=0.8836713744616165,
            initial_failure=0.5,
            duplicate_update="group_mean",
            curvature_refresh_interval=20,
            price_proxy_scale=1.0842683683055117,
        )


@dataclass(frozen=True)
class FenchelBregmanOutput:
    loss: Tensor
    failure: Tensor
    weight: Tensor


def failure_surrogate(per_example_loss: Tensor, *, temperature: float) -> Tensor:
    """Map non-negative task loss to the verified smooth failure coordinate."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if per_example_loss.ndim != 1 or per_example_loss.numel() == 0:
        raise ValueError("per_example_loss must be a non-empty vector")
    if per_example_loss.is_complex() or not torch.isfinite(per_example_loss).all():
        raise ValueError("per_example_loss must be finite and real")
    if bool((per_example_loss < 0).any()):
        raise ValueError("per_example_loss must be non-negative")
    return -torch.expm1(-per_example_loss / float(temperature))


def fenchel_bregman_objective(
    per_example_loss: Tensor,
    failure_probability: Tensor,
    *,
    config: FenchelBregmanConfig,
) -> FenchelBregmanOutput:
    """Return the verified failure-weighted primal gradient objective."""

    if failure_probability.shape != per_example_loss.shape:
        raise ValueError("failure probability and task loss shapes must match")
    if failure_probability.is_complex() or not torch.isfinite(
        failure_probability
    ).all():
        raise ValueError("failure_probability must be finite and real")
    if bool(((failure_probability < 0) | (failure_probability > 1)).any()):
        raise ValueError("failure_probability must lie in [0, 1]")
    failure = failure_surrogate(
        per_example_loss, temperature=config.temperature
    )
    estimate = failure_probability.detach().to(
        device=failure.device, dtype=failure.dtype
    )
    weight = (
        float(config.candidate_count)
        * estimate.pow(config.candidate_count - 1)
    ).detach()
    loss = (float(config.gradient_scale) * weight * failure).sum()
    return FenchelBregmanOutput(loss=loss, failure=failure, weight=weight)


def should_refresh_curvature(step: int, interval: int | None) -> bool:
    if isinstance(step, bool) or int(step) != step or step < 0:
        raise ValueError("step must be a non-negative integer")
    if interval is None:
        return False
    if isinstance(interval, bool) or int(interval) != interval or interval < 1:
        raise ValueError("interval must be a positive integer")
    return int(step) % int(interval) == 0


def price_anchored_curvature_gradient(
    mean_gradient: Tensor,
    *,
    ordinary_gradient: Tensor | None,
    step: int,
    config: FenchelBregmanConfig,
) -> tuple[Tensor, bool]:
    """Choose the verified Sudoku IVON curvature gradient for one step."""

    if config.price_proxy_scale is None or config.curvature_refresh_interval is None:
        raise ValueError("price-anchored curvature requires a proxy scale and interval")
    if mean_gradient.ndim != 1 or not torch.isfinite(mean_gradient).all():
        raise ValueError("mean_gradient must be a finite vector")
    refreshed = should_refresh_curvature(step, config.curvature_refresh_interval)
    if refreshed:
        if ordinary_gradient is None:
            raise ValueError("an ordinary gradient is required at a curvature anchor")
        if ordinary_gradient.shape != mean_gradient.shape or not torch.isfinite(
            ordinary_gradient
        ).all():
            raise ValueError("ordinary_gradient must be a matching finite vector")
        return ordinary_gradient, True
    if ordinary_gradient is not None:
        raise ValueError("ordinary_gradient is only accepted at a curvature anchor")
    return float(config.price_proxy_scale) * mean_gradient, False


class FailureProbabilityTracker:
    """Visited-only EMA failure state with deterministic duplicate ordering."""

    def __init__(
        self,
        num_groups: int,
        *,
        beta: float,
        initial_failure: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(num_groups, bool) or int(num_groups) != num_groups or num_groups < 1:
            raise ValueError("num_groups must be positive")
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if not math.isfinite(initial_failure) or not 0 <= initial_failure <= 1:
            raise ValueError("initial_failure must lie in [0, 1]")
        if not dtype.is_floating_point:
            raise ValueError("tracker dtype must be floating point")
        self.num_groups = int(num_groups)
        self.beta = float(beta)
        self.initial_failure = float(initial_failure)
        self.values = torch.full(
            (self.num_groups,), self.initial_failure, device=device, dtype=dtype
        )
        self.visit_counts = torch.zeros(
            self.num_groups, device=device, dtype=torch.int64
        )
        self._weight_cache: dict[int, Tensor] = {}

    def _indices(self, group_ids: Tensor) -> Tensor:
        if not torch.is_tensor(group_ids) or group_ids.ndim != 1:
            raise ValueError("group_ids must be a one-dimensional tensor")
        raw = group_ids.detach().to(device=self.values.device)
        if raw.dtype == torch.bool or raw.is_complex():
            raise ValueError("group_ids must contain integers")
        indices = raw.to(torch.long)
        if raw.is_floating_point() and not torch.equal(raw, indices.to(raw.dtype)):
            raise ValueError("group_ids must contain integers")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.num_groups
        ):
            raise IndexError("group id is outside tracker range")
        return indices

    def values_for(self, group_ids: Tensor) -> Tensor:
        indices = self._indices(group_ids)
        return self.values[indices].detach().to(device=group_ids.device)

    def weights_for(self, group_ids: Tensor, *, candidate_count: int) -> Tensor:
        if (
            isinstance(candidate_count, bool)
            or int(candidate_count) != candidate_count
            or candidate_count < 2
        ):
            raise ValueError("candidate_count must be an integer of at least two")
        values = self.values_for(group_ids)
        cached = self._weight_cache.get(int(candidate_count))
        if cached is not None:
            return cached[self._indices(group_ids)].detach().to(device=group_ids.device)
        return (float(candidate_count) * values.pow(candidate_count - 1)).detach()

    @torch.no_grad()
    def update(
        self,
        group_ids: Tensor,
        failures: Tensor,
        *,
        duplicate_update: str = "sequential",
    ) -> None:
        indices = self._indices(group_ids)
        observed = failures.detach().to(
            device=self.values.device, dtype=self.values.dtype
        )
        if observed.ndim != 1 or observed.shape != indices.shape:
            raise ValueError("group_ids and failures must be matching vectors")
        if observed.is_complex() or not torch.isfinite(observed).all():
            raise ValueError("failures must be finite and real")
        if bool(((observed < 0) | (observed > 1)).any()):
            raise ValueError("failures must lie in [0, 1]")
        if duplicate_update == "group_mean":
            value_order = torch.argsort(observed, stable=True)
            order = value_order[torch.argsort(indices[value_order], stable=True)]
            sorted_indices = indices[order]
            sorted_values = observed[order]
            unique, counts = torch.unique_consecutive(
                sorted_indices, return_counts=True
            )
            indices = unique
            observed = torch.stack(
                [segment.mean() for segment in sorted_values.split(counts.tolist())]
            )
        elif duplicate_update != "sequential":
            raise ValueError("duplicate_update must be sequential or group_mean")
        # Sequential mode preserves rank-major gathered visit order. Group-mean
        # mode reproduces the verified Sudoku dual update.
        for index, value in zip(indices.tolist(), observed, strict=True):
            self.values[index].mul_(self.beta).add_(value, alpha=1 - self.beta)
            self.visit_counts[index].add_(1)
            for candidate_count, cached in self._weight_cache.items():
                cached[index] = float(candidate_count) * self.values[index].pow(
                    candidate_count - 1
                )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "RRM_FAILURE_TRACKER_V1",
            "num_groups": self.num_groups,
            "beta": self.beta,
            "initial_failure": self.initial_failure,
            "values": self.values.detach().clone(),
            "visit_counts": self.visit_counts.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") == "RRM_FAILURE_TRACKER_V1":
            if (
                state.get("num_groups") != self.num_groups
                or state.get("beta") != self.beta
                or state.get("initial_failure") != self.initial_failure
            ):
                raise ValueError("failure tracker metadata mismatch")
            values = state.get("values")
            visits = state.get("visit_counts")
        elif state.get("format_version") == 2:
            if (
                state.get("num_groups") != self.num_groups
                or state.get("mode") != "failure_ema"
                or state.get("ema_beta") != self.beta
                or state.get("initial_failure") != self.initial_failure
            ):
                raise ValueError("legacy Fenchel failure tracker metadata mismatch")
            values = state.get("failure_estimate")
            visits = state.get("update_count")
        else:
            if state.get("beta") != self.beta:
                raise ValueError("legacy failure tracker beta mismatch")
            values = state.get("values")
            visits = state.get("visit_counts")
        if not isinstance(values, Tensor) or not isinstance(visits, Tensor):
            raise ValueError("failure tracker state is missing tensors")
        if values.shape != self.values.shape or visits.shape != self.visit_counts.shape:
            raise ValueError("failure tracker tensor shape mismatch")
        candidate_values = values.to(device=self.values.device, dtype=self.values.dtype)
        candidate_visits = visits.to(device=self.visit_counts.device, dtype=torch.int64)
        if not torch.isfinite(candidate_values).all() or bool(
            ((candidate_values < 0) | (candidate_values > 1)).any()
        ):
            raise ValueError("failure tracker values must lie in [0, 1]")
        if bool((candidate_visits < 0).any()):
            raise ValueError("failure tracker visit counts must be non-negative")
        self.values.copy_(candidate_values)
        self.visit_counts.copy_(candidate_visits)
        self._weight_cache.clear()
        if state.get("format_version") == 2:
            candidate_count = state.get("k")
            dual = state.get("dual")
            if (
                isinstance(candidate_count, int)
                and candidate_count >= 2
                and isinstance(dual, Tensor)
                and dual.shape == self.values.shape
            ):
                self._weight_cache[candidate_count] = dual.to(
                    device=self.values.device, dtype=self.values.dtype
                ).clone()
