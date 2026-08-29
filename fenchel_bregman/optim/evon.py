"""Adapter around the official GPLv3+ EVON package."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

import torch
from torch import Tensor

from evon import EVON as _UpstreamEVON


class EVON(_UpstreamEVON):
    """Official EVON with validation, scaled sampling, and lean checkpoints."""

    optimizer_name = "evon"

    def __init__(
        self,
        *args: Any,
        hess_clip_ratio: float | None = None,
        price_clip_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        hess_init = float(kwargs.get("hess_init", args[2] if len(args) > 2 else 0.0))
        if hess_init <= 0:
            raise ValueError(f"hess_init must be positive, got {hess_init}")
        frequency = int(kwargs.get("precondition_frequency", 10))
        if frequency <= 0:
            raise ValueError("precondition_frequency must be positive")
        max_dim = int(kwargs.get("max_precond_dim", 10000))
        if max_dim <= 0:
            raise ValueError("max_precond_dim must be positive")
        if hess_clip_ratio is not None and price_clip_ratio is not None:
            if float(hess_clip_ratio) != float(price_clip_ratio):
                raise ValueError("hess_clip_ratio and price_clip_ratio disagree")
        clip_ratio = price_clip_ratio if price_clip_ratio is not None else hess_clip_ratio
        if clip_ratio is not None and clip_ratio <= 0:
            raise ValueError("Hessian clip ratio must be positive when provided")
        super().__init__(*args, price_clip_ratio=clip_ratio, **kwargs)
        self.profile_timing_enabled = False
        self.profile_eigenspace_update_s = 0.0
        self.profile_eigenspace_update_count = 0

    def reset_profile_timings(self) -> None:
        self.profile_eigenspace_update_s = 0.0
        self.profile_eigenspace_update_count = 0

    def _update_preconditioner(self, *args: Any, **kwargs: Any) -> None:
        if not self.profile_timing_enabled:
            return super()._update_preconditioner(*args, **kwargs)
        parameter = args[0]
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
        start = time.perf_counter()
        result = super()._update_preconditioner(*args, **kwargs)
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
        self.profile_eigenspace_update_s += time.perf_counter() - start
        self.profile_eigenspace_update_count += 1
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore upstream EVON state in its configured work dtype."""
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
    def sampled_params_scaled(self, *, train: bool, scale: float = 1.0):
        """Sample with a temporary posterior standard-deviation multiplier."""
        if scale < 0:
            raise ValueError("posterior scale must be non-negative")
        if scale == 0:
            yield
            return
        if scale == 1:
            with self.sampled_params(train=train):
                yield
            return

        original_ess = [group["ess"] for group in self.param_groups]
        try:
            for group, ess in zip(self.param_groups, original_ess):
                group["ess"] = ess / (scale * scale)
            with self.sampled_params(train=train):
                yield
        finally:
            for group, ess in zip(self.param_groups, original_ess):
                group["ess"] = ess

    def state_dict(self) -> dict[str, Any]:
        """Exclude the reconstructible posterior-mean scratch buffers."""
        payload = super().state_dict()
        payload["state"] = {
            key: (
                {name: value for name, value in state.items() if name != "_mean_buf"}
                if isinstance(state, dict)
                else state
            )
            for key, state in payload["state"].items()
        }
        return payload
