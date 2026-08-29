from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from ivon import IVON as _UpstreamIVON


UpdateTransform = Literal["clip", "none", "muon_whiten"]


def _expected_standard_normal_max(sample_count: int) -> float:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if sample_count == 1:
        return 0.0
    grid = torch.linspace(-8.0, 8.0, steps=16001, dtype=torch.float64)
    normal_pdf = torch.exp(-0.5 * grid.square()) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + torch.erf(grid / math.sqrt(2.0)))
    integrand = sample_count * grid * normal_pdf * normal_cdf.pow(sample_count - 1)
    return float(torch.trapezoid(integrand, grid))


class IVON(_UpstreamIVON):
    """IVON with an explicit update-transform switch.

    The default ``update_transform="clip"`` follows upstream IVON's clipped
    preconditioned update and is the reproduction path for the existing runs.
    Other values are intentionally opt-in hooks for optimizer-internal
    experiments.
    """

    _MUON_EXCLUDED_NAME_FRAGMENTS = (
        "embed",
        "lm_head",
        "q_head",
    )

    def __init__(
        self,
        *args,
        update_transform: UpdateTransform = "clip",
        muon_whiten_eps: float = 1e-8,
        muon_ns_steps: int = 5,
        pdr_curvature_lambda: float = 0.0,
        pdr_target_k: int = 10,
        pdr_curvature_eps: float = 1e-12,
        pdr_curvature_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        if update_transform not in ("clip", "none", "muon_whiten"):
            raise ValueError(f"Unsupported IVON update_transform: {update_transform}")
        if muon_whiten_eps <= 0:
            raise ValueError("muon_whiten_eps must be positive.")
        if muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive.")
        if pdr_curvature_lambda < 0.0:
            raise ValueError("pdr_curvature_lambda must be non-negative.")
        if pdr_target_k < 1:
            raise ValueError("pdr_target_k must be positive.")
        if pdr_curvature_eps <= 0.0:
            raise ValueError("pdr_curvature_eps must be positive.")
        if not 0.0 <= pdr_curvature_ratio < 1.0:
            raise ValueError("pdr_curvature_ratio must be in [0, 1).")
        if pdr_curvature_lambda > 0.0 and pdr_curvature_ratio > 0.0:
            raise ValueError("Choose either pdr_curvature_lambda or pdr_curvature_ratio, not both.")
        self.update_transform = update_transform
        self.muon_whiten_eps = float(muon_whiten_eps)
        self.muon_ns_steps = int(muon_ns_steps)
        self.pdr_curvature_lambda = float(pdr_curvature_lambda)
        self.pdr_target_k = int(pdr_target_k)
        self.pdr_curvature_eps = float(pdr_curvature_eps)
        self.pdr_curvature_ratio = float(pdr_curvature_ratio)
        self.pdr_a_k = _expected_standard_normal_max(self.pdr_target_k)
        self.pdr_last_diagnostics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _sample_params(self) -> tuple[Tensor, Tensor]:
        if not any("posterior_std_override" in group for group in self.param_groups):
            return super()._sample_params()

        noise_samples = []
        param_avgs = []
        offset = 0
        for group in self.param_groups:
            std = group.get("posterior_std_override")
            if std is None:
                std = 1.0 / (
                    group["ess"] * (group["hess"] + group["weight_decay"])
                ).sqrt()
            std = std.to(device=self._device, dtype=self._dtype).flatten()
            if std.numel() != group["numel"] or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("posterior_std_override must be finite, positive, and match the IVON group size.")
            noise_sample = torch.randn(
                group["numel"], device=self._device, dtype=self._dtype
            ) * std
            noise_samples.append(noise_sample)
            group_offset = 0
            for parameter in group["params"]:
                if parameter is None:
                    continue
                param_avg = parameter.data.flatten()
                numel = parameter.numel()
                param_noise = noise_sample[group_offset : group_offset + numel]
                param_avgs.append(param_avg)
                parameter.data = (param_avg + param_noise).view(parameter.shape)
                group_offset += numel
                offset += numel
            assert group_offset == group["numel"]
        assert offset == self._numel
        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _transform_update(self, update: Tensor, clip_radius: float, group: dict | None = None) -> Tensor:
        if self.update_transform == "clip":
            return torch.clip(update, min=-clip_radius, max=clip_radius)
        if self.update_transform == "none":
            return update
        if group is None:
            raise ValueError("muon_whiten requires an optimizer parameter group.")
        return self._transform_muon_group_update(update, clip_radius, group)

    def _transform_muon_group_update(self, update: Tensor, clip_radius: float, group: dict) -> Tensor:
        pieces = []
        names = group.get("param_names")
        group_offset = 0
        param_index = 0
        for param in group["params"]:
            if param is None:
                continue
            param_update = update[group_offset : group_offset + param.numel()].view(param.shape)
            param_name = names[param_index] if names is not None and param_index < len(names) else None
            if self._should_muon_whiten(param, param_name):
                pieces.append(self._muon_orthogonalize(param_update).reshape(-1))
            else:
                pieces.append(torch.clip(param_update, min=-clip_radius, max=clip_radius).reshape(-1))
            group_offset += param.numel()
            param_index += 1
        assert group_offset == group["numel"]
        return torch.cat(pieces, 0)

    def _should_muon_whiten(self, param: Tensor, param_name: str | None) -> bool:
        if param.ndim != 2 or min(param.shape) <= 1:
            return False
        if param_name is None:
            return True
        normalized = param_name.removeprefix("_orig_mod.")
        return not any(fragment in normalized for fragment in self._MUON_EXCLUDED_NAME_FRAGMENTS)

    def _muon_orthogonalize(self, update: Tensor) -> Tensor:
        if not torch.isfinite(update).all():
            return update
        x = update.to(torch.float32)
        should_transpose = x.shape[0] > x.shape[1]
        if should_transpose:
            x = x.T

        x = x / x.norm().clamp_min(self.muon_whiten_eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(self.muon_ns_steps):
            xx_t = x @ x.T
            x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

        if should_transpose:
            x = x.T
        x = x * math.sqrt(max(1.0, update.shape[0] / update.shape[1]))
        return x.to(dtype=update.dtype)

    def _update(self) -> None:
        self.current_step += 1

        correction_sq_sum = 0.0
        correction_count = 0
        floor_count = 0
        s_sum = 0.0
        group_count = 0

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice],
                group["momentum"],
                b1,
            )

            if group.get("freeze_hess", False):
                pass
            elif self.pdr_curvature_lambda == 0.0 and self.pdr_curvature_ratio == 0.0:
                group["hess"] = self._new_hess(
                    self.hess_approx,
                    group["hess"],
                    self.state["avg_nxg"],
                    self.state["avg_gsq"],
                    pg_slice,
                    group["ess"],
                    b2,
                    group["weight_decay"],
                )
            elif self.pdr_curvature_ratio > 0.0:
                base_hess = group.get("pdr_base_hess")
                if base_hess is None:
                    base_hess = group["hess"].detach().clone()
                base_hess = self._new_hess(
                    self.hess_approx,
                    base_hess,
                    self.state["avg_nxg"],
                    self.state["avg_gsq"],
                    pg_slice,
                    group["ess"],
                    b2,
                    group["weight_decay"],
                )
                group["pdr_base_hess"] = base_hess
                gradient = self.state["avg_grad"][pg_slice]
                precision_floor = max(
                    self.pdr_curvature_eps,
                    float(torch.finfo(base_hess.dtype).eps),
                )
                precision = (base_hess + group["weight_decay"]).clamp_min(precision_floor)
                variance = 1.0 / (group["ess"] * precision)
                uncertainty_sq = (variance * gradient.square()).sum() + self.pdr_curvature_eps
                raw_correction = self.pdr_a_k * gradient.square() / torch.sqrt(uncertainty_sq)
                base_hess_rms = torch.sqrt(base_hess.square().mean() + self.pdr_curvature_eps)
                raw_correction_rms = torch.sqrt(raw_correction.square().mean() + self.pdr_curvature_eps)
                target_correction_rms = self.pdr_curvature_ratio * base_hess_rms
                correction = raw_correction * (target_correction_rms / raw_correction_rms)
                minimum_hess = -group["weight_decay"] + precision_floor
                coordinate_cap = 0.1 * (base_hess - minimum_hess).clamp_min(0.0)
                correction = torch.minimum(correction, coordinate_cap).clamp_min(0.0)
                corrected_hess = torch.clamp(base_hess - correction, min=minimum_hess)
                floor_count += int((corrected_hess <= minimum_hess).sum().cpu())
                correction_sq_sum += float(correction.float().square().sum().cpu())
                correction_count += correction.numel()
                s_sum += float(torch.sqrt(uncertainty_sq).cpu())
                group_count += 1
                group["hess"] = corrected_hess
            else:
                base_hess = group.get("pdr_base_hess")
                if base_hess is None:
                    base_hess = group["hess"].detach().clone()
                base_hess = self._new_hess(
                    self.hess_approx,
                    base_hess,
                    self.state["avg_nxg"],
                    self.state["avg_gsq"],
                    pg_slice,
                    group["ess"],
                    b2,
                    group["weight_decay"],
                )
                group["pdr_base_hess"] = base_hess
                gradient = self.state["avg_grad"][pg_slice]
                precision_floor = max(
                    self.pdr_curvature_eps,
                    float(torch.finfo(base_hess.dtype).eps),
                )
                precision = (base_hess + group["weight_decay"]).clamp_min(precision_floor)
                variance = 1.0 / (group["ess"] * precision)
                uncertainty = torch.sqrt((variance * gradient.square()).sum() + self.pdr_curvature_eps)
                correction = self.pdr_curvature_lambda * self.pdr_a_k * gradient.square() / uncertainty
                minimum_hess = -group["weight_decay"] + precision_floor
                corrected_hess = torch.clamp(base_hess - correction, min=minimum_hess)
                floor_count += int((corrected_hess <= minimum_hess).sum().cpu())
                correction_sq_sum += float(correction.float().square().sum().cpu())
                correction_count += correction.numel()
                s_sum += float(uncertainty.cpu())
                group_count += 1
                group["hess"] = corrected_hess

            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0
            lr_scale = lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr
            raw_update = (group["momentum"] / debias + group["weight_decay"] * param_avg) / (
                group["hess"] + group["weight_decay"]
            )
            param_avg = param_avg - lr_scale * self._transform_update(raw_update, group["clip_radius"], group)

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel
        if self.pdr_curvature_lambda > 0.0 or self.pdr_curvature_ratio > 0.0:
            self.pdr_last_diagnostics = {
                "pdr/curvature_s": s_sum / max(group_count, 1),
                "pdr/curvature_correction_rms": math.sqrt(correction_sq_sum / max(correction_count, 1)),
                "pdr/curvature_floor_fraction": floor_count / max(correction_count, 1),
                "pdr/curvature_a_k": self.pdr_a_k,
            }
        else:
            self.pdr_last_diagnostics = {}

    @torch.no_grad()
    def step_from_weighted_samples(
        self,
        losses: list[Tensor],
        gradients: list[Tensor],
        noises: list[Tensor],
        *,
        alpha: float,
        temperature: float,
    ) -> tuple[Tensor, Tensor]:
        if not losses or len(losses) != len(gradients) or len(losses) != len(noises):
            raise ValueError("PDR samples must have matching non-empty loss, gradient, and noise lists.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"PDR alpha must be in [0, 1], got {alpha}.")
        if temperature <= 0.0:
            raise ValueError(f"PDR temperature must be positive, got {temperature}.")

        loss_tensor = torch.stack([loss.detach().to(self._device, dtype=torch.float32) for loss in losses])
        sample_count = loss_tensor.numel()
        softmin_weights = torch.softmax(-loss_tensor / temperature, dim=0)
        weights = (1.0 - alpha) / sample_count + alpha * softmin_weights

        avg_grad = torch.zeros_like(gradients[0])
        for weight, gradient in zip(weights, gradients, strict=True):
            avg_grad.add_(gradient, alpha=float(weight))
        self.state["count"] = sample_count
        self.state["avg_grad"] = avg_grad
        if self.hess_approx == "price":
            avg_nxg = torch.zeros_like(noises[0])
            for weight, noise, gradient in zip(weights, noises, gradients, strict=True):
                avg_nxg.add_(noise * gradient, alpha=float(weight))
            self.state["avg_nxg"] = avg_nxg
        else:
            avg_gsq = torch.zeros_like(gradients[0])
            for weight, gradient in zip(weights, gradients, strict=True):
                avg_gsq.add_(gradient.square(), alpha=float(weight))
            self.state["avg_gsq"] = avg_gsq

        self._update()
        self._reset_samples()
        mean_loss = loss_tensor.mean()
        softmin_loss = -temperature * (
            torch.logsumexp(-loss_tensor / temperature, dim=0) - math.log(sample_count)
        )
        objective = (1.0 - alpha) * mean_loss + alpha * softmin_loss
        return objective, weights

    @torch.no_grad()
    def step_from_preweighted_samples(
        self,
        gradients: list[Tensor],
        noises: list[Tensor],
        *,
        objective: Tensor,
    ) -> Tensor:
        if not gradients or len(gradients) != len(noises):
            raise ValueError("PDR samples must have matching non-empty gradient and noise lists.")

        avg_grad = torch.zeros_like(gradients[0])
        for gradient in gradients:
            avg_grad.add_(gradient)
        self.state["count"] = len(gradients)
        self.state["avg_grad"] = avg_grad
        if self.hess_approx == "price":
            avg_nxg = torch.zeros_like(noises[0])
            for noise, gradient in zip(noises, gradients, strict=True):
                avg_nxg.add_(noise * gradient)
            self.state["avg_nxg"] = avg_nxg
        else:
            avg_gsq = torch.zeros_like(gradients[0])
            for gradient in gradients:
                avg_gsq.add_(gradient.square())
            self.state["avg_gsq"] = avg_gsq

        self._update()
        self._reset_samples()
        return objective.detach().to(self._device, dtype=torch.float32)

    @torch.no_grad()
    def step_from_split_gradient(
        self,
        mean_gradient: Tensor,
        curvature_gradient: Tensor,
        noise: Tensor,
        *,
        objective: Tensor,
    ) -> Tensor:
        if mean_gradient.shape != curvature_gradient.shape or mean_gradient.shape != noise.shape:
            raise ValueError("Split IVON gradients and noise must have matching shapes.")
        self.state["count"] = 1
        self.state["avg_grad"] = mean_gradient
        if self.hess_approx == "price":
            self.state["avg_nxg"] = noise * curvature_gradient
        else:
            self.state["avg_gsq"] = curvature_gradient.square()
        self._update()
        self._reset_samples()
        return objective.detach().to(self._device, dtype=torch.float32)

    @torch.no_grad()
    def step_from_factorial_samples(
        self,
        gradients: list[Tensor],
        noises: list[Tensor],
        *,
        mean_sample_indices: tuple[int, ...],
        precision_sample_indices: tuple[int, ...],
        objective: Tensor,
    ) -> Tensor:
        if len(gradients) != 2 or len(noises) != 2:
            raise ValueError("The registered IVON factorial requires exactly two samples.")
        if any(index not in (0, 1) for index in (*mean_sample_indices, *precision_sample_indices)):
            raise ValueError("Factorial sample indices must be 0 or 1.")
        if not mean_sample_indices or not precision_sample_indices:
            raise ValueError("Factorial sample-index sets must be non-empty.")
        if any(gradient.shape != gradients[0].shape for gradient in gradients):
            raise ValueError("Factorial gradients must have identical shapes.")
        if any(noise.shape != gradients[0].shape for noise in noises):
            raise ValueError("Factorial noises and gradients must have identical shapes.")

        self.state["count"] = 2
        self.state["avg_grad"] = torch.stack(
            [gradients[index] for index in mean_sample_indices]
        ).mean(0)
        if self.hess_approx == "price":
            self.state["avg_nxg"] = torch.stack(
                [noises[index] * gradients[index] for index in precision_sample_indices]
            ).mean(0)
        else:
            self.state["avg_gsq"] = torch.stack(
                [gradients[index].square() for index in precision_sample_indices]
            ).mean(0)
        self._update()
        self._reset_samples()
        return objective.detach().to(self._device, dtype=torch.float32)

    @torch.no_grad()
    def step_from_utility_samples(
        self,
        mean_gradients: list[Tensor],
        precision_gradients: list[Tensor],
        precision_noises: list[Tensor],
        *,
        objective: Tensor,
    ) -> Tensor:
        if len(mean_gradients) not in (2, 4):
            raise ValueError("Finite-K utility mean gradients require K in {2, 4}.")
        if len(precision_gradients) != 2 or len(precision_noises) != 2:
            raise ValueError("Finite-K utility requires exactly two ordinary precision samples.")
        shape = mean_gradients[0].shape
        if any(gradient.shape != shape for gradient in (*mean_gradients, *precision_gradients)):
            raise ValueError("Finite-K utility gradients must have identical shapes.")
        if any(noise.shape != shape for noise in precision_noises):
            raise ValueError("Finite-K utility precision noises must match gradient shapes.")

        self.state["count"] = 2
        self.state["avg_grad"] = torch.stack(mean_gradients).sum(0)
        if self.hess_approx == "price":
            self.state["avg_nxg"] = torch.stack([
                noise * gradient
                for noise, gradient in zip(precision_noises, precision_gradients, strict=True)
            ]).mean(0)
        else:
            self.state["avg_gsq"] = torch.stack([
                gradient.square() for gradient in precision_gradients
            ]).mean(0)
        self._update()
        self._reset_samples()
        return objective.detach().to(self._device, dtype=torch.float32)
