"""Paper-derived GRAM model for discrete Sudoku reasoning.

The implementation follows equations (4)--(9) and the truncated surrogate in
equation (14) of *Generative Recursive Reasoning*.  Choices not fixed by the
paper are deliberately exposed in :class:`GRAMConfig` and documented in the
experiment assumption register.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def rms_norm(value: Tensor, epsilon: float) -> Tensor:
    """Parameter-free RMS normalization used by the TRM Sudoku backbone."""
    original_dtype = value.dtype
    normalized = value.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(original_dtype)


class CastedLinear(nn.Module):
    """Float32 parameters evaluated in the activation dtype."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.trunc_normal_(self.weight, std=in_features**-0.5, a=-2 * in_features**-0.5, b=2 * in_features**-0.5)

    def forward(self, value: Tensor) -> Tensor:
        bias = self.bias.to(value.dtype) if self.bias is not None else None
        return F.linear(value, self.weight.to(value.dtype), bias)


class SwiGLU(nn.Module):
    """SwiGLU projection with the same width rounding convention as TRM."""

    def __init__(self, hidden_size: int, expansion: float, *, multiple: int = 256) -> None:
        super().__init__()
        intermediate = _round_up(round(expansion * hidden_size * 2 / 3), multiple)
        self.gate_up = CastedLinear(hidden_size, 2 * intermediate)
        self.down = CastedLinear(intermediate, hidden_size)

    def forward(self, value: Tensor) -> Tensor:
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SudokuRecursiveBlock(nn.Module):
    """The paper's Sudoku-specific ``SwiGLU + SwiGLU`` recursive block."""

    def __init__(
        self,
        sequence_size: int,
        hidden_size: int,
        expansion: float,
        epsilon: float,
        width_multiple: int,
    ) -> None:
        super().__init__()
        self.token_mixer = SwiGLU(sequence_size, expansion, multiple=width_multiple)
        self.channel_mixer = SwiGLU(hidden_size, expansion, multiple=width_multiple)
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        transposed = hidden.transpose(1, 2)
        transposed = rms_norm(transposed + self.token_mixer(transposed), self.epsilon)
        hidden = transposed.transpose(1, 2)
        return rms_norm(hidden + self.channel_mixer(hidden), self.epsilon)


class RotaryPositionEmbedding(nn.Module):
    """Rotary position encoding for attention queries and keys."""

    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 10_000.0,
        style: str = "interleaved",
    ) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE head dimension must be a positive even number")
        if style not in ("interleaved", "half"):
            raise ValueError("RoPE style must be interleaved or half")
        self.style = style
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length,
            device=query.device,
            dtype=self.inverse_frequency.dtype,
        )
        angles = torch.outer(positions, self.inverse_frequency)
        if self.style == "half":
            cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
            sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
        else:
            cosine = angles.cos()
            sine = angles.sin()
        cosine = cosine.to(dtype=query.dtype).view(1, 1, sequence_length, -1)
        sine = sine.to(dtype=query.dtype).view(1, 1, sequence_length, -1)

        def rotate(value: Tensor) -> Tensor:
            if self.style == "half":
                first, second = value.chunk(2, dim=-1)
                rotated = torch.cat((-second, first), dim=-1)
                return value * cosine + rotated * sine
            even = value[..., 0::2]
            odd = value[..., 1::2]
            rotated_even = even * cosine - odd * sine
            rotated_odd = even * sine + odd * cosine
            return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

        return rotate(query), rotate(key)


class AttentionRecursiveBlock(nn.Module):
    """ARC-style recursive block: self-attention followed by a SwiGLU MLP."""

    def __init__(
        self,
        sequence_size: int,
        hidden_size: int,
        expansion: float,
        epsilon: float,
        width_multiple: int,
        attention_heads: int,
        position_encoding: str,
        rope_style: str = "interleaved",
    ) -> None:
        super().__init__()
        if attention_heads < 1 or hidden_size % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_size")
        head_dim = hidden_size // attention_heads
        if position_encoding not in ("none", "rope"):
            raise ValueError("position_encoding must be none or rope")
        self.sequence_size = sequence_size
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.head_dim = head_dim
        self.position_encoding = position_encoding
        self.query_key_value = CastedLinear(hidden_size, 3 * hidden_size)
        self.output_projection = CastedLinear(hidden_size, hidden_size)
        self.rotary = (
            RotaryPositionEmbedding(head_dim, style=rope_style)
            if position_encoding == "rope"
            else None
        )
        self.channel_mixer = SwiGLU(hidden_size, expansion, multiple=width_multiple)
        self.epsilon = epsilon

    def _attention(self, hidden: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden.shape
        query, key, value = self.query_key_value(hidden).chunk(3, dim=-1)

        def heads(value: Tensor) -> Tensor:
            return value.view(batch_size, sequence_length, self.attention_heads, self.head_dim).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        if self.rotary is not None:
            query, key = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)
        return self.output_projection(attended)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = rms_norm(hidden + self._attention(hidden), self.epsilon)
        return rms_norm(hidden + self.channel_mixer(hidden), self.epsilon)


class SharedRecursiveCore(nn.Module):
    """Weight-shared core used for both low- and high-level updates."""

    def __init__(self, config: "GRAMConfig") -> None:
        super().__init__()
        sequence_size = config.sequence_length + config.register_tokens
        block_type = SudokuRecursiveBlock if config.core_architecture == "sudoku_dense" else AttentionRecursiveBlock
        self.layers = nn.ModuleList(
            block_type(
                sequence_size,
                config.hidden_size,
                config.core_expansion,
                config.rms_norm_eps,
                config.width_multiple,
                **(
                {
                    "attention_heads": config.attention_heads,
                    "position_encoding": config.position_encoding,
                    "rope_style": config.rope_style,
                }
                    if config.core_architecture == "attention"
                    else {}
                ),
            )
            for _ in range(config.core_layers)
        )

    def forward(self, state: Tensor, injection: Tensor) -> Tensor:
        state = state + injection
        for layer in self.layers:
            state = layer(state)
        return state


class GuidanceDistribution(nn.Module):
    """State-dependent diagonal Gaussian for a GRAM prior or posterior."""

    def __init__(self, config: "GRAMConfig") -> None:
        super().__init__()
        self.mean = SwiGLU(
            config.hidden_size, config.guidance_expansion, multiple=config.width_multiple
        )
        self.scale = SwiGLU(
            config.hidden_size, config.guidance_expansion, multiple=config.width_multiple
        )
        self.minimum_std = config.minimum_std
        self.maximum_std = config.maximum_std

    def forward(self, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean(conditioning)
        std = F.softplus(self.scale(conditioning)) + self.minimum_std
        if self.maximum_std is not None:
            std = std.clamp_max(self.maximum_std)
        return mean, std


@dataclass(frozen=True)
class GRAMConfig:
    sequence_length: int = 81
    vocab_size: int = 11
    hidden_size: int = 512
    register_tokens: int = 16
    core_layers: int = 2
    low_level_steps: int = 6
    transitions_per_supervision: int = 3
    supervision_steps: int = 16
    core_expansion: float = 4.0
    guidance_expansion: float = 1.0
    decoder_expansion: float = 4.0
    width_multiple: int = 256
    rms_norm_eps: float = 1e-5
    minimum_std: float = 1e-4
    maximum_std: float | None = None
    forward_dtype: str = "bfloat16"
    core_architecture: str = "sudoku_dense"
    attention_heads: int = 8
    position_encoding: str = "none"
    rope_style: str = "interleaved"
    puzzle_embedding_tokens: int = 0
    share_recursive_core: bool = True
    positionwise_initial_state: bool = True
    decoder_type: str = "swiglu"
    guidance_mode: str = "gaussian"

    def __post_init__(self) -> None:
        positive_ints = (
            "sequence_length",
            "vocab_size",
            "hidden_size",
            "core_layers",
            "low_level_steps",
            "transitions_per_supervision",
            "supervision_steps",
            "width_multiple",
            "attention_heads",
        )
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.register_tokens < 1:
            raise ValueError("register_tokens must be positive for halt/LPRM readout")
        if self.core_architecture not in ("sudoku_dense", "attention"):
            raise ValueError("core_architecture must be sudoku_dense or attention")
        if self.position_encoding not in ("none", "rope"):
            raise ValueError("position_encoding must be none or rope")
        if self.rope_style not in ("interleaved", "half"):
            raise ValueError("rope_style must be interleaved or half")
        if self.decoder_type not in ("swiglu", "identity"):
            raise ValueError("decoder_type must be swiglu or identity")
        if self.guidance_mode not in ("gaussian", "zero"):
            raise ValueError("guidance_mode must be gaussian or zero")
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_size")
        if not 0 <= self.puzzle_embedding_tokens <= self.register_tokens:
            raise ValueError("puzzle_embedding_tokens must lie between 0 and register_tokens")
        for name in ("core_expansion", "guidance_expansion", "decoder_expansion", "rms_norm_eps", "minimum_std"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_std is not None and self.maximum_std <= self.minimum_std:
            raise ValueError("maximum_std must exceed minimum_std")
        if self.forward_dtype not in ("bfloat16", "float32"):
            raise ValueError("forward_dtype must be bfloat16 or float32")

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.forward_dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GRAMConfig":
        return cls(**payload)


@dataclass
class GRAMLatentState:
    high: Tensor
    low: Tensor

    def detach(self) -> "GRAMLatentState":
        return GRAMLatentState(self.high.detach(), self.low.detach())


@dataclass
class GRAMTrainingOutput:
    state: GRAMLatentState
    loss: Tensor
    logits: Tensor
    value: Tensor
    halt_logits: Tensor
    metrics: dict[str, Tensor]


@dataclass
class GRAMCandidates:
    predictions: Tensor
    values: Tensor
    logits: Tensor | None = None


def diagonal_gaussian_kl(q_mean: Tensor, q_std: Tensor, p_mean: Tensor, p_std: Tensor) -> Tensor:
    """Elementwise ``KL(N(q)||N(p))`` for diagonal Gaussian distributions."""
    variance_ratio = q_std.square() / p_std.square()
    mean_term = (q_mean - p_mean).square() / p_std.square()
    return torch.log(p_std / q_std) + 0.5 * (variance_ratio + mean_term - 1.0)


def balanced_gaussian_kl(
    q_mean: Tensor,
    q_std: Tensor,
    p_mean: Tensor,
    p_std: Tensor,
    balance: float,
) -> Tensor:
    """Dreamer-style KL balancing used for the paper's coefficient of 0.8."""
    if not 0.0 <= balance <= 1.0:
        raise ValueError("KL balance must lie in [0, 1]")
    prior_update = diagonal_gaussian_kl(q_mean.detach(), q_std.detach(), p_mean, p_std)
    posterior_update = diagonal_gaussian_kl(q_mean, q_std, p_mean.detach(), p_std.detach())
    return balance * prior_update + (1.0 - balance) * posterior_update


def stablemax_cross_entropy(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Per-token StableMax cross entropy used by the Sudoku TRM training stack."""
    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, 0).long()
    values = logits.double()
    transformed = torch.where(values < 0, 1 / (1 - values + 1e-30), values + 1)
    log_probabilities = torch.log(transformed / transformed.sum(dim=-1, keepdim=True))
    selected = torch.gather(log_probabilities, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    # Keep the scalar reduction in float64, as in the validated TRM loss. The
    # gradient is cast back at the BF16 logits boundary by autograd.
    return -torch.where(valid, selected, 0)


def weighted_token_loss(
    token_losses: Tensor,
    labels: Tensor,
    *,
    route_token_weight: float = 1.0,
    route_token: int = 5,
    ignore_index: int = -100,
) -> Tensor:
    """Average per-example token loss with optional Maze route-marker weighting."""
    if route_token_weight <= 0:
        raise ValueError("route_token_weight must be positive")
    valid = labels != ignore_index
    weights = torch.where(
        labels.eq(route_token),
        torch.as_tensor(route_token_weight, device=labels.device, dtype=token_losses.dtype),
        torch.ones_like(token_losses),
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=-1).clamp_min(1)
    return (token_losses * weights).sum(dim=-1).div(denominator).mean()


def route_connectivity_loss(logits: Tensor, labels: Tensor, *, route_token: int = 5) -> Tensor:
    """Supervise local route adjacency around target path cells."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have compatible [batch, sequence, ...] shapes")
    sequence_length = labels.shape[1]
    side = math.isqrt(sequence_length)
    if side * side != sequence_length:
        raise ValueError("route connectivity requires a square sequence")
    route_logits = logits[..., [3, 4, route_token]]
    non_route_logits = torch.cat((logits[..., :3], logits[..., 6:]), dim=-1)
    route_logit = torch.logsumexp(route_logits, dim=-1) - torch.logsumexp(non_route_logits, dim=-1)
    route_target = labels.eq(route_token) | labels.eq(3) | labels.eq(4)
    score = route_logit.view(-1, side, side)
    target = route_target.view(-1, side, side)
    padded_score = F.pad(score, (1, 1, 1, 1), value=0.0)
    padded_target = F.pad(target, (1, 1, 1, 1), value=False)
    padded_valid = F.pad(torch.ones_like(target, dtype=torch.bool), (1, 1, 1, 1), value=False)
    neighbor_scores = torch.stack(
        (
            padded_score[:, :-2, 1:-1],
            padded_score[:, 2:, 1:-1],
            padded_score[:, 1:-1, :-2],
            padded_score[:, 1:-1, 2:],
        ),
        dim=-1,
    )
    neighbor_targets = torch.stack(
        (
            padded_target[:, :-2, 1:-1],
            padded_target[:, 2:, 1:-1],
            padded_target[:, 1:-1, :-2],
            padded_target[:, 1:-1, 2:],
        ),
        dim=-1,
    )
    neighbor_valid = torch.stack(
        (
            padded_valid[:, :-2, 1:-1],
            padded_valid[:, 2:, 1:-1],
            padded_valid[:, 1:-1, :-2],
            padded_valid[:, 1:-1, 2:],
        ),
        dim=-1,
    )
    valid = target.unsqueeze(-1).expand_as(neighbor_valid) & neighbor_valid
    if not valid.any():
        return logits.new_zeros((), dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(
        neighbor_scores[valid], neighbor_targets[valid].float(), reduction="mean"
    )


def reduce_gaussian_kl(elementwise: Tensor, reduction: str) -> Tensor:
    """Reduce a diagonal-state KL while retaining its latent-vector semantics."""
    if reduction == "element_mean":
        return elementwise.mean()
    if reduction == "sum_hidden_mean_tokens":
        return elementwise.sum(dim=-1).mean()
    raise ValueError(f"Unsupported KL reduction: {reduction}")


class GenerativeRecursiveReasoningModel(nn.Module):
    """Hierarchical GRAM for Sudoku and ARC-style structured sequences."""

    def __init__(self, config: GRAMConfig) -> None:
        super().__init__()
        self.config = config
        embed_std = config.hidden_size**-0.5
        self.token_embedding = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        nn.init.trunc_normal_(self.token_embedding, std=embed_std, a=-2 * embed_std, b=2 * embed_std)
        self.puzzle_embedding = nn.Parameter(
            torch.empty(config.puzzle_embedding_tokens, config.hidden_size)
        )
        if config.puzzle_embedding_tokens:
            nn.init.trunc_normal_(self.puzzle_embedding, std=embed_std, a=-2 * embed_std, b=2 * embed_std)

        self.low_core = SharedRecursiveCore(config)
        self.high_core = self.low_core if config.share_recursive_core else SharedRecursiveCore(config)
        self.prior = GuidanceDistribution(config)
        self.posterior = GuidanceDistribution(config)
        self.posterior_conditioner = CastedLinear(2 * config.hidden_size, config.hidden_size)
        self.decoder = (
            SwiGLU(config.hidden_size, config.decoder_expansion, multiple=config.width_multiple)
            if config.decoder_type == "swiglu"
            else nn.Identity()
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size)
        self.halt_head = CastedLinear(config.hidden_size, 1, bias=True)
        self.value_head = CastedLinear(config.hidden_size, 1, bias=True)

        initial_shape = (
            (config.sequence_length + config.register_tokens, config.hidden_size)
            if config.positionwise_initial_state
            else (config.hidden_size,)
        )
        initial_high = torch.empty(initial_shape, dtype=config.torch_dtype)
        initial_low = torch.empty(initial_shape, dtype=config.torch_dtype)
        # Appendix B.1 specifies one draw from N(0, I), persisted with the
        # checkpoint so every trajectory starts from the same fixed state.
        nn.init.normal_(initial_high, mean=0.0, std=1.0)
        nn.init.normal_(initial_low, mean=0.0, std=1.0)
        self.register_buffer("initial_high", initial_high, persistent=True)
        self.register_buffer("initial_low", initial_low, persistent=True)

        with torch.no_grad():
            self.halt_head.weight.zero_()
            assert self.halt_head.bias is not None
            self.halt_head.bias.fill_(-5.0)

    @property
    def sequence_size(self) -> int:
        return self.config.sequence_length + self.config.register_tokens

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(self, batch_size: int, device: torch.device | str) -> GRAMLatentState:
        shape = (batch_size, self.sequence_size, self.config.hidden_size)
        if self.config.positionwise_initial_state:
            high = self.initial_high.to(device=device).view(1, self.sequence_size, -1).expand(shape).clone()
            low = self.initial_low.to(device=device).view(1, self.sequence_size, -1).expand(shape).clone()
        else:
            high = self.initial_high.to(device=device).view(1, 1, -1).expand(shape).clone()
            low = self.initial_low.to(device=device).view(1, 1, -1).expand(shape).clone()
        return GRAMLatentState(high=high, low=low)

    def reset_state(self, state: GRAMLatentState, reset: Tensor) -> GRAMLatentState:
        if reset.ndim != 1 or reset.shape[0] != state.high.shape[0]:
            raise ValueError("reset mask must have shape [batch]")
        view = reset.view(-1, 1, 1)
        initial = self.initial_state(state.high.shape[0], state.high.device)
        return GRAMLatentState(
            high=torch.where(view, initial.high, state.high),
            low=torch.where(view, initial.low, state.low),
        )

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"tokens must have shape [batch, {self.config.sequence_length}], got {tuple(tokens.shape)}"
            )
        content = F.embedding(tokens.long(), self.token_embedding.to(self.config.torch_dtype))
        content = math.sqrt(self.config.hidden_size) * content
        registers = torch.zeros(
            tokens.shape[0],
            self.config.register_tokens,
            self.config.hidden_size,
            device=tokens.device,
            dtype=self.config.torch_dtype,
        )
        if self.config.puzzle_embedding_tokens:
            puzzle = self.puzzle_embedding.to(self.config.torch_dtype)
            registers[:, : self.config.puzzle_embedding_tokens] = math.sqrt(self.config.hidden_size) * puzzle
        return torch.cat((registers, content), dim=1)

    def _decode(self, high: Tensor) -> Tensor:
        content = high[:, self.config.register_tokens :]
        return self.lm_head(self.decoder(content))

    def _posterior_conditioning(self, labels: Tensor, fusion: str) -> Tensor:
        target = self._embed_tokens(labels.clamp_min(0))
        if fusion == "tokenwise":
            return target
        if fusion == "global_token_mixer":
            # Reuse the Sudoku token mixer as a position-aware target encoder,
            # then expose only one global vector to every posterior position.
            # This preserves q(epsilon | u, y) without allowing a label token
            # to be copied directly into its matching output position.
            first_layer = self.low_core.layers[0]
            if hasattr(first_layer, "token_mixer"):
                mixed = first_layer.token_mixer(target.transpose(1, 2))
                global_target = mixed[:, :, 0]
            else:
                mixed = first_layer(target)
                global_target = mixed[:, 0, :]
            return global_target.unsqueeze(1).expand(-1, self.sequence_size, -1)
        if fusion == "mean_pool_concat":
            return target.mean(dim=1, keepdim=True)
        raise ValueError(f"Unsupported posterior fusion: {fusion}")

    def _sample(
        self,
        mean: Tensor,
        std: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + std * noise

    def _transition(
        self,
        state: GRAMLatentState,
        input_embedding: Tensor,
        target_embedding: Tensor | None,
        generator: torch.Generator | None,
        *,
        kl_balance: float | None,
        kl_reduction: str = "sum_hidden_mean_tokens",
    ) -> tuple[GRAMLatentState, Tensor | None, Tensor | None]:
        low = state.low
        for _ in range(self.config.low_level_steps):
            low = self.low_core(low, state.high + input_embedding)
        proposal = self.high_core(state.high, low)
        prior_mean, prior_std = self.prior(proposal)

        kl: Tensor | None = None
        sampled_std_mean: Tensor = prior_std.detach().float().mean()
        if target_embedding is None:
            guidance = (
                self._sample(prior_mean, prior_std, generator)
                if self.config.guidance_mode == "gaussian"
                else torch.zeros_like(proposal)
            )
        else:
            if target_embedding.shape[1] == 1:
                target = target_embedding.expand(-1, proposal.shape[1], -1)
                posterior_input = self.posterior_conditioner(torch.cat((proposal, target), dim=-1))
            else:
                posterior_input = proposal + target_embedding
            posterior_mean, posterior_std = self.posterior(posterior_input)
            sampled_std_mean = posterior_std.detach().float().mean()
            guidance = (
                self._sample(posterior_mean, posterior_std, generator)
                if self.config.guidance_mode == "gaussian"
                else torch.zeros_like(proposal)
            )
            if kl_balance is not None:
                kl = balanced_gaussian_kl(
                    posterior_mean,
                    posterior_std,
                    prior_mean,
                    prior_std,
                    kl_balance,
                )
                kl = reduce_gaussian_kl(kl, kl_reduction)
        return GRAMLatentState(high=proposal + guidance, low=low), kl, sampled_std_mean

    def training_supervision(
        self,
        state: GRAMLatentState,
        inputs: Tensor,
        labels: Tensor,
        *,
        beta: float,
        kl_balance: float,
        halt_loss_weight: float,
        lprm_loss_weight: float,
        route_token_weight: float = 1.0,
        route_auxiliary_loss_weight: float = 0.0,
        route_auxiliary_pos_weight: float = 1.0,
        route_connectivity_loss_weight: float = 0.0,
        token_loss_type: str = "stablemax_cross_entropy",
        kl_reduction: str = "sum_hidden_mean_tokens",
        latent_source: str = "posterior",
        posterior_fusion: str = "global_token_mixer",
        generator: torch.Generator | None = None,
    ) -> GRAMTrainingOutput:
        """Run one deep-supervision segment with only its final transition tracked."""
        if (
            beta < 0
            or halt_loss_weight < 0
            or lprm_loss_weight < 0
            or route_auxiliary_loss_weight < 0
            or route_auxiliary_pos_weight <= 0
            or route_connectivity_loss_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")
        input_embedding = self._embed_tokens(inputs)
        valid = labels != -100
        valid_count = valid.sum(dim=-1).clamp_min(1)

        def rollout(target_embedding: Tensor | None) -> tuple[GRAMLatentState, Tensor, Tensor, list[Tensor]]:
            value_states: list[Tensor] = []
            current = state.detach()
            for _ in range(self.config.transitions_per_supervision - 1):
                with torch.no_grad():
                    current, _, _ = self._transition(
                        current,
                        input_embedding,
                        target_embedding,
                        generator,
                        kl_balance=None,
                        kl_reduction=kl_reduction,
                    )
                value_states.append(current.high.detach())
            current, kl, sampled_std_mean = self._transition(
                current.detach(),
                input_embedding,
                target_embedding,
                generator,
                kl_balance=kl_balance if target_embedding is not None else None,
                kl_reduction=kl_reduction,
            )
            if kl is None:
                kl = current.high.new_zeros((), dtype=torch.float32)
            value_states.append(current.high.detach())
            return current, kl, sampled_std_mean, value_states

        def branch_losses(
            logits: Tensor,
            value_states: list[Tensor],
            current: GRAMLatentState,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            if token_loss_type == "stablemax_cross_entropy":
                token_losses = stablemax_cross_entropy(logits, labels, ignore_index=-100)
            elif token_loss_type == "softmax_cross_entropy":
                token_losses = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    labels.long().reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels)
            else:
                raise ValueError(f"Unsupported token loss: {token_loss_type}")
            token_loss = weighted_token_loss(
                token_losses,
                labels,
                route_token_weight=route_token_weight,
            )
            if route_auxiliary_loss_weight > 0:
                non_route_logits = torch.cat((logits[..., :5], logits[..., 6:]), dim=-1)
                route_logit = logits[..., 5] - torch.logsumexp(non_route_logits, dim=-1)
                route_target = labels.eq(5).float()
                route_valid = valid
                route_auxiliary_loss = F.binary_cross_entropy_with_logits(
                    route_logit[route_valid],
                    route_target[route_valid],
                    pos_weight=torch.as_tensor(
                        route_auxiliary_pos_weight,
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                )
            else:
                route_auxiliary_loss = logits.new_zeros((), dtype=torch.float32)
            connectivity_loss = (
                route_connectivity_loss(logits, labels)
                if route_connectivity_loss_weight > 0
                else logits.new_zeros((), dtype=torch.float32)
            )
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = valid & predictions.eq(labels)
                token_accuracy = correct.sum(dim=-1).float() / valid_count.float()
                exact = correct.sum(dim=-1).eq(valid_count)
            halt_logits = self.halt_head(current.high.detach()[:, 0]).squeeze(-1).float()
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits, exact.float())
            values = torch.stack(
                [self.value_head(high[:, 0]).squeeze(-1).float() for high in value_states],
                dim=1,
            )
            lprm_loss = F.mse_loss(values, token_accuracy[:, None].expand_as(values))
            return (
                token_loss,
                token_accuracy,
                exact,
                halt_logits,
                halt_loss,
                lprm_loss,
                route_auxiliary_loss,
                connectivity_loss,
            )

        if latent_source in ("posterior", "prior"):
            target_embedding = (
                self._posterior_conditioning(labels, posterior_fusion)
                if latent_source == "posterior"
                else None
            )
            current, kl, sampled_std_mean, value_states = rollout(target_embedding)
            logits = self._decode(current.high)
            token_loss, token_accuracy, exact, halt_logits, halt_loss, lprm_loss, route_auxiliary_loss, connectivity_loss = branch_losses(
                logits, value_states, current
            )
            total_loss = (
                token_loss
                + beta * kl
                + halt_loss_weight * halt_loss
                + lprm_loss_weight * lprm_loss
                + route_auxiliary_loss_weight * route_auxiliary_loss
                + route_connectivity_loss_weight * connectivity_loss
            )
            metrics = {
                "loss": total_loss.detach(),
                "token_loss": token_loss.detach(),
                "kl": kl.detach(),
                "halt_loss": halt_loss.detach(),
                "lprm_loss": lprm_loss.detach(),
                "route_auxiliary_loss": route_auxiliary_loss.detach(),
                "route_connectivity_loss": connectivity_loss.detach(),
                "token_accuracy": token_accuracy.mean().detach(),
                "exact_accuracy": exact.float().mean().detach(),
                "guidance_std_mean": sampled_std_mean,
                "posterior_std_mean": sampled_std_mean,
            }
            return GRAMTrainingOutput(
                state=current.detach(),
                loss=total_loss,
                logits=logits,
                value=self.value_head(value_states[-1][:, 0]).squeeze(-1).float(),
                halt_logits=halt_logits,
                metrics=metrics,
            )

        if latent_source != "dual":
            raise ValueError(f"Unsupported training latent source: {latent_source}")

        target_embedding = self._posterior_conditioning(labels, posterior_fusion)
        prior_current, _, prior_std, prior_values = rollout(None)
        posterior_current, kl, posterior_std, posterior_values = rollout(target_embedding)
        prior_logits = self._decode(prior_current.high)
        posterior_logits = self._decode(posterior_current.high)
        prior = branch_losses(prior_logits, prior_values, prior_current)
        posterior = branch_losses(posterior_logits, posterior_values, posterior_current)
        (
            prior_token_loss,
            prior_token_accuracy,
            prior_exact,
            prior_halt_logits,
            prior_halt_loss,
            prior_lprm_loss,
            prior_route_auxiliary_loss,
            prior_connectivity_loss,
        ) = prior
        (
            posterior_token_loss,
            posterior_token_accuracy,
            posterior_exact,
            _,
            posterior_halt_loss,
            posterior_lprm_loss,
            posterior_route_auxiliary_loss,
            posterior_connectivity_loss,
        ) = posterior
        token_loss = 0.5 * (prior_token_loss + posterior_token_loss)
        halt_loss = 0.5 * (prior_halt_loss + posterior_halt_loss)
        lprm_loss = 0.5 * (prior_lprm_loss + posterior_lprm_loss)
        route_auxiliary_loss = 0.5 * (prior_route_auxiliary_loss + posterior_route_auxiliary_loss)
        connectivity_loss = 0.5 * (prior_connectivity_loss + posterior_connectivity_loss)
        total_loss = (
            token_loss
            + beta * kl
            + halt_loss_weight * halt_loss
            + lprm_loss_weight * lprm_loss
            + route_auxiliary_loss_weight * route_auxiliary_loss
            + route_connectivity_loss_weight * connectivity_loss
        )
        metrics = {
            "loss": total_loss.detach(),
            "token_loss": token_loss.detach(),
            "kl": kl.detach(),
            "halt_loss": halt_loss.detach(),
            "lprm_loss": lprm_loss.detach(),
            "route_auxiliary_loss": route_auxiliary_loss.detach(),
            "route_connectivity_loss": connectivity_loss.detach(),
            "token_accuracy": prior_token_accuracy.mean().detach(),
            "exact_accuracy": prior_exact.float().mean().detach(),
            "prior_token_accuracy": prior_token_accuracy.mean().detach(),
            "posterior_token_accuracy": posterior_token_accuracy.mean().detach(),
            "prior_exact_accuracy": prior_exact.float().mean().detach(),
            "posterior_exact_accuracy": posterior_exact.float().mean().detach(),
            "prior_guidance_std_mean": prior_std,
            "guidance_std_mean": posterior_std,
            "posterior_std_mean": posterior_std,
        }
        return GRAMTrainingOutput(
            state=prior_current.detach(),
            loss=total_loss,
            logits=prior_logits,
            value=self.value_head(prior_values[-1][:, 0]).squeeze(-1).float(),
            halt_logits=prior_halt_logits,
            metrics=metrics,
        )

    @torch.no_grad()
    def infer_candidates(
        self,
        inputs: Tensor,
        *,
        samples: int,
        supervision_steps: int | None = None,
        generator: torch.Generator | None = None,
        return_logits: bool = False,
    ) -> GRAMCandidates:
        """Sample parallel prior trajectories and return terminal LPRM values."""
        if samples < 1:
            raise ValueError("samples must be positive")
        steps = self.config.supervision_steps if supervision_steps is None else supervision_steps
        if steps < 1:
            raise ValueError("supervision_steps must be positive")
        batch_size = inputs.shape[0]
        expanded_inputs = inputs.repeat_interleave(samples, dim=0)
        input_embedding = self._embed_tokens(expanded_inputs)
        state = self.initial_state(expanded_inputs.shape[0], inputs.device)
        for _ in range(steps):
            for _ in range(self.config.transitions_per_supervision):
                state, _, _ = self._transition(
                    state,
                    input_embedding,
                    target_embedding=None,
                    generator=generator,
                    kl_balance=None,
                    kl_reduction="sum_hidden_mean_tokens",
                )
            state = state.detach()
        logits = self._decode(state.high)
        predictions = logits.argmax(dim=-1).view(batch_size, samples, self.config.sequence_length)
        values = self.value_head(state.high[:, 0]).squeeze(-1).float().view(batch_size, samples)
        shaped_logits = (
            logits.view(batch_size, samples, self.config.sequence_length, self.config.vocab_size)
            if return_logits
            else None
        )
        return GRAMCandidates(predictions=predictions, values=values, logits=shaped_logits)

from pathlib import Path
from typing import Mapping

from torch.optim import Optimizer

from rrm.utils import EvaluationOutput, TrainingOutput, normalize_state_dict_keys, sha256_file


def gram_config_from_payload(payload: Mapping[str, Any]) -> GRAMConfig:
    """Build the exact model config, including documented legacy defaults."""

    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("GRAM checkpoint does not contain config metadata")
    raw_model = raw_config.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("GRAM checkpoint config does not contain a model section")
    values = dict(raw_model)
    # Checkpoints created before these fields were serialized used a single
    # broadcast initial vector and one shared recursive core.
    values.setdefault("positionwise_initial_state", False)
    values.setdefault("share_recursive_core", True)
    return GRAMConfig.from_dict(values)


def _gram_compatible_state(
    model: GenerativeRecursiveReasoningModel,
    state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    normalized = dict(
        normalize_state_dict_keys(
            state,
            prefixes=("_orig_mod.model.", "_orig_mod.", "model."),
        )
    )
    if "puzzle_embedding" not in normalized and model.config.puzzle_embedding_tokens == 0:
        normalized["puzzle_embedding"] = model.puzzle_embedding.detach().clone()
    # The released Sudoku checkpoint predates the posterior fusion projection.
    # It is unused by the prior-only inference path that produced the table.
    if "posterior_conditioner.weight" not in normalized:
        normalized["posterior_conditioner.weight"] = (
            model.posterior_conditioner.weight.detach().clone()
        )
    if not any(key.startswith("low_core.") for key in normalized):
        for key in list(normalized):
            if key.startswith("core."):
                normalized["low_core." + key[len("core."):]] = normalized.pop(key)
    if model.config.share_recursive_core and not any(
        key.startswith("high_core.") for key in normalized
    ):
        for key, value in list(normalized.items()):
            if key.startswith("low_core."):
                normalized["high_core." + key[len("low_core."):]] = value
    return normalized


def load_gram_weights(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    use_ema: bool,
) -> str:
    """Strict-load the requested point or EMA weights and return the state key."""

    if not isinstance(model, GenerativeRecursiveReasoningModel):
        raise TypeError("GRAM weights require GenerativeRecursiveReasoningModel")
    if use_ema:
        if isinstance(payload.get("ema_model_state_dict"), Mapping):
            key = "ema_model_state_dict"
            state = payload[key]
        elif isinstance(payload.get("ema_state_dict"), Mapping):
            ema = payload["ema_state_dict"]
            shadow = ema.get("shadow") if isinstance(ema, Mapping) else None
            if not isinstance(shadow, Mapping):
                raise ValueError("GRAM EMA state does not contain shadow weights")
            key = "ema_state_dict"
            state = shadow
        else:
            raise ValueError("GRAM checkpoint does not contain EMA weights")
    else:
        if isinstance(payload.get("model_state_dict"), Mapping):
            key = "model_state_dict"
            state = payload[key]
        elif isinstance(payload.get("model_state"), Mapping):
            key = "model_state"
            state = payload[key]
        else:
            raise ValueError("GRAM checkpoint does not contain point weights")
    model.load_state_dict(_gram_compatible_state(model, state), strict=True)
    return key


def select_gram_candidates(
    predictions: Tensor, values: Tensor
) -> tuple[Tensor, Tensor]:
    """Select the first terminal-value maximum for every puzzle."""

    if predictions.ndim != 3 or values.ndim != 2:
        raise ValueError("predictions and values must have [B, K, L] and [B, K] shapes")
    if predictions.shape[:2] != values.shape:
        raise ValueError("prediction and value candidate axes differ")
    if not torch.isfinite(values).all():
        raise ValueError("candidate values must be finite")
    indices = values.argmax(dim=1)
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    return predictions[rows, indices], values[rows, indices]


def transfer_trm_weights(
    model: GenerativeRecursiveReasoningModel,
    source_state: Mapping[str, Tensor],
) -> dict[str, list[str]]:
    """Map the deterministic TRM backbone and heads into a GRAM warm start."""

    source = dict(
        normalize_state_dict_keys(
            source_state,
            prefixes=("_orig_mod.model.", "_orig_mod.", "model."),
        )
    )
    target = model.state_dict()
    mapped_targets: set[str] = set()
    used_sources: set[str] = set()

    def copy(target_key: str, source_key: str, *, transform=None) -> None:
        if source_key not in source:
            raise ValueError(f"TRM warm start is missing {source_key}")
        value = source[source_key]
        if transform is not None:
            value = transform(value)
        if target_key not in target:
            raise ValueError(f"GRAM warm-start target is missing {target_key}")
        if value.shape != target[target_key].shape:
            raise ValueError(
                f"Warm-start shape mismatch for {target_key}: "
                f"{tuple(value.shape)} versus {tuple(target[target_key].shape)}"
            )
        target[target_key] = value.detach().to(dtype=target[target_key].dtype).clone()
        mapped_targets.add(target_key)
        used_sources.add(source_key)

    copy("token_embedding", "inner.embed_tokens.embedding_weight")
    copy("initial_high", "inner.H_init")
    copy("initial_low", "inner.L_init")
    if model.config.puzzle_embedding_tokens:
        copy(
            "puzzle_embedding",
            "inner.puzzle_emb.weights",
            transform=lambda value: value[0, : model.puzzle_embedding.numel()].reshape_as(
                model.puzzle_embedding
            ),
        )
    layer_mapping = {
        "query_key_value.weight": "self_attn.qkv_proj.weight",
        "output_projection.weight": "self_attn.o_proj.weight",
        "channel_mixer.gate_up.weight": "mlp.gate_up_proj.weight",
        "channel_mixer.down.weight": "mlp.down_proj.weight",
    }
    for layer_index in range(model.config.core_layers):
        for target_suffix, source_suffix in layer_mapping.items():
            source_key = f"inner.L_level.layers.{layer_index}.{source_suffix}"
            copy(f"low_core.layers.{layer_index}.{target_suffix}", source_key)
            copy(f"high_core.layers.{layer_index}.{target_suffix}", source_key)
    copy("lm_head.weight", "inner.lm_head.weight")
    copy(
        "halt_head.weight",
        "inner.q_head.weight",
        transform=lambda value: value[0:1],
    )
    copy(
        "halt_head.bias",
        "inner.q_head.bias",
        transform=lambda value: value[0:1],
    )
    copy(
        "value_head.weight",
        "inner.q_head.weight",
        transform=lambda value: value[0:1],
    )
    copy(
        "value_head.bias",
        "inner.q_head.bias",
        transform=lambda value: value[0:1],
    )
    model.load_state_dict(target, strict=True)
    initialized_targets = sorted(set(target).difference(mapped_targets))
    return {
        "mapped_target": sorted(mapped_targets),
        "initialized_target": initialized_targets,
        "used_source": sorted(used_sources),
        "unaccounted_source": sorted(set(source).difference(used_sources)),
        "unaccounted_target": [],
    }


def _per_example_gram_token_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    token_loss_type: str,
    route_token_weight: float,
) -> Tensor:
    if token_loss_type == "stablemax_cross_entropy":
        losses = stablemax_cross_entropy(logits, labels, ignore_index=-100)
    elif token_loss_type == "softmax_cross_entropy":
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.long().reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
    else:
        raise ValueError(f"Unsupported token loss: {token_loss_type}")
    valid = labels != -100
    weights = torch.where(
        labels.eq(5),
        torch.as_tensor(route_token_weight, device=labels.device, dtype=losses.dtype),
        torch.ones_like(losses),
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return (losses * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)


class GRAMAdapter:
    """Translate GRAM latent supervision and value selection to RRM contracts."""

    name = "gram"

    def build_model(
        self, task: str, preset: str, metadata: Mapping[str, Any]
    ) -> GenerativeRecursiveReasoningModel:
        if task not in ("maze-hard", "sudoku-extreme"):
            raise ValueError(f"GRAM does not support task {task!r}")
        raw = metadata.get("model", metadata.get("arch", metadata))
        if not isinstance(raw, Mapping):
            raise ValueError("GRAM metadata must contain a model config")
        model = GenerativeRecursiveReasoningModel(GRAMConfig.from_dict(dict(raw)))
        model._rrm_candidate_count = int(metadata.get("candidate_count", 10))
        model._rrm_supervision_steps = int(
            metadata.get("supervision_steps", model.config.supervision_steps)
        )
        if model._rrm_candidate_count < 1 or model._rrm_supervision_steps < 1:
            raise ValueError("candidate_count and supervision_steps must be positive")
        model._rrm_training_options = {
            "beta": float(metadata.get("kl_beta", 0.0)),
            "kl_balance": float(metadata.get("kl_balance", 0.8)),
            "halt_loss_weight": float(metadata.get("halt_loss_weight", 1.0)),
            "lprm_loss_weight": float(metadata.get("lprm_loss_weight", 1.0)),
            "route_token_weight": float(metadata.get("route_token_loss_weight", 1.0)),
            "route_auxiliary_loss_weight": float(
                metadata.get("route_auxiliary_loss_weight", 0.0)
            ),
            "route_auxiliary_pos_weight": float(
                metadata.get("route_auxiliary_pos_weight", 1.0)
            ),
            "route_connectivity_loss_weight": float(
                metadata.get("route_connectivity_loss_weight", 0.0)
            ),
            "token_loss_type": str(
                metadata.get("token_loss_type", "stablemax_cross_entropy")
            ),
            "kl_reduction": str(
                metadata.get("kl_reduction", "sum_hidden_mean_tokens")
            ),
            "latent_source": str(metadata.get("latent_source", "posterior")),
            "posterior_fusion": str(
                metadata.get("posterior_fusion", "global_token_mixer")
            ),
        }
        return model

    def load_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> Mapping[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("GRAM checkpoint must contain a mapping")
        if payload.get("format") == "RRM_CHECKPOINT_V1":
            adapter_state = payload.get("adapter_state")
            ema_state = (
                adapter_state.get("ema_model_state")
                if isinstance(adapter_state, Mapping)
                else None
            )
            state = ema_state if isinstance(ema_state, Mapping) else payload.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError("RRM GRAM checkpoint is missing model weights")
            model.load_state_dict(_gram_compatible_state(model, state), strict=True)
        else:
            load_gram_weights(model, payload, use_ema=True)
        return payload

    def training_forward(
        self,
        model: nn.Module,
        state: object,
        batch: Mapping[str, Tensor],
    ) -> TrainingOutput:
        if not isinstance(model, GenerativeRecursiveReasoningModel):
            raise TypeError("GRAMAdapter requires GenerativeRecursiveReasoningModel")
        batch_size = int(batch["inputs"].shape[0])
        if state is None:
            initial = model.initial_state(batch_size, batch["inputs"].device)
            state = {
                "high": initial.high,
                "low": initial.low,
                "halted": torch.ones(
                    batch_size, dtype=torch.bool, device=batch["inputs"].device
                ),
                "steps": torch.zeros(
                    batch_size, dtype=torch.long, device=batch["inputs"].device
                ),
                "inputs": torch.zeros_like(batch["inputs"]),
                "labels": torch.zeros_like(batch["labels"]),
            }
        if not isinstance(state, Mapping) or any(
            not isinstance(state.get(name), Tensor)
            for name in ("high", "low", "halted", "steps", "inputs", "labels")
        ):
            raise TypeError("GRAM training state must contain the registered carry tensors")
        halted = state["halted"]
        if halted.shape != (batch_size,):
            raise ValueError("GRAM carry batch size differs from the candidate batch")
        active_inputs = torch.where(
            halted.unsqueeze(1), batch["inputs"], state["inputs"]
        )
        active_labels = torch.where(
            halted.unsqueeze(1), batch["labels"], state["labels"]
        )
        latent = model.reset_state(
            GRAMLatentState(state["high"], state["low"]), halted
        )
        options = dict(getattr(model, "_rrm_training_options", {}))
        native = model.training_supervision(
            latent,
            active_inputs,
            active_labels,
            **options,
        )
        task_loss = _per_example_gram_token_loss(
            native.logits,
            active_labels,
            token_loss_type=str(options.get("token_loss_type", "stablemax_cross_entropy")),
            route_token_weight=float(options.get("route_token_weight", 1.0)),
        )
        auxiliary = batch_size * (native.loss - task_loss.mean())
        steps = torch.where(halted, 0, state["steps"]) + 1
        next_state = {
            "high": native.state.high.detach(),
            "low": native.state.low.detach(),
            "halted": steps.ge(model.config.supervision_steps),
            "steps": steps,
            "inputs": active_inputs,
            "labels": active_labels,
        }
        return TrainingOutput(
            state=next_state,
            per_example_task_loss=task_loss,
            auxiliary_loss=auxiliary,
            predictions=native.logits.argmax(dim=-1),
            selection_score=native.value,
            metrics=dict(native.metrics),
        )

    @torch.no_grad()
    def evaluation_candidates(
        self,
        model: nn.Module,
        batch: Mapping[str, Tensor],
        *,
        candidate_count: int | None = None,
        supervision_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(model, GenerativeRecursiveReasoningModel):
            raise TypeError("GRAMAdapter requires GenerativeRecursiveReasoningModel")
        was_training = model.training
        model.eval()
        candidates = int(
            getattr(model, "_rrm_candidate_count", 10)
            if candidate_count is None
            else candidate_count
        )
        steps = int(
            getattr(model, "_rrm_supervision_steps", model.config.supervision_steps)
            if supervision_steps is None
            else supervision_steps
        )
        native = model.infer_candidates(
            batch["inputs"],
            samples=candidates,
            supervision_steps=steps,
            generator=generator,
        )
        if was_training:
            model.train()
        return native.predictions, native.values

    @torch.no_grad()
    def evaluation_forward(
        self, model: nn.Module, batch: Mapping[str, Tensor]
    ) -> EvaluationOutput:
        if not isinstance(model, GenerativeRecursiveReasoningModel):
            raise TypeError("GRAMAdapter requires GenerativeRecursiveReasoningModel")
        candidate_predictions, candidate_scores = self.evaluation_candidates(
            model, batch
        )
        predictions, scores = select_gram_candidates(
            candidate_predictions, candidate_scores
        )
        steps = int(
            getattr(model, "_rrm_supervision_steps", model.config.supervision_steps)
        )
        batch_size = int(batch["inputs"].shape[0])
        device = batch["inputs"].device
        return EvaluationOutput(
            predictions=predictions,
            selection_score=scores,
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            steps=torch.full((batch_size,), steps, dtype=torch.int32, device=device),
        )

    def build_optimizers(
        self, model: nn.Module, preset: Mapping[str, Any]
    ) -> tuple[Optimizer, Optimizer | None]:
        dense = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(preset.get("learning_rate", preset.get("lr", 1e-4))),
            weight_decay=float(preset.get("weight_decay", 0.0)),
            betas=(
                float(preset.get("beta1", 0.9)),
                float(preset.get("beta2", 0.95)),
            ),
            foreach=False,
            fused=False,
        )
        return dense, None


ADAPTER = GRAMAdapter()
