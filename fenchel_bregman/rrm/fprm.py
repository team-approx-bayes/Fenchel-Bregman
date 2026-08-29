"""Standalone fixed-point reasoning model.

The model implementation is consolidated from the upstream FPRM repository at
the pinned revision below.  Its model equations and checkpoint-visible names
are intentionally preserved for released-checkpoint compatibility.

MIT License

Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
Copyright (c) 2026 Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin,
Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch,
Antonio Orvieto

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
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import einops
from pydantic import BaseModel
from scipy.stats import expon, gamma
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention
from torch.optim.optimizer import Optimizer, ParamsT

from rrm.utils import EvaluationOutput, TrainingOutput, normalize_state_dict_keys


FPRM_UPSTREAM_URL = "https://github.com/nilskiKonjIzDunava/fprm"
FPRM_UPSTREAM_COMMIT = "4fd7ab116b1361c4fb040b26b270f19a3d5319b4"


# ---- Consolidated upstream component 1 ----

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

# ---- Consolidated upstream component 2 ----

class ReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    n_backwards_L: int = 1   # number of with-grad L-level steps in the final pass; was implicitly L_cycles before

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Q-head input source: True -> read from z_H[:, 0] (first puzzle_emb register, original behavior).
    # False -> mean-pool z_H over the non-puzzle_emb token positions.
    q_logit_from_puzzle_emb: bool = True
    # When True, q-head loss only updates the q-head itself, not the reasoning trunk.
    q_logit_detach_model: bool = False

    # When True, self-attention uses a causal mask (left-to-right). Default False
    # for non-autoregressive tasks (ARC, sudoku, maze); set True for tasks like
    # state tracking where prefix-k accuracy from a single forward is desired.
    causal: bool = False

    # Fields introduced by FPTRM.
    # Defaults reproduce the original TRM block: post-norm, no conv, no residual scaling.
    softmax_temp: float = 1.0
    norm_type: str = "post-norm"            # 'pre-norm' | 'peri-norm' | 'post-norm'
    norm_placement: str = "none"            # 'input' | 'output' | 'none'
    conv_type: str = "none"                 # 'conv1d' | 'conv2d' | 'none'
    conv_kernel_size: int = 4
    conv_bias: bool = False
    residual_scale: Optional[str] = None    # None | 'fixed' | 'input-independent' | 'input-dependent'
    alpha_1_init: float = 0.5
    alpha_2_init: float = 0.5
    normalize_input_injection: bool = False
    use_spec_norm_linear: bool = False

    # When True, only alpha_2 (the outer input-injection mix) is learnable;
    # the per-block alpha_1/beta_1 are pinned to 1 so blocks behave as plain residuals.
    outer_only: bool = False

    # DropConnect-style weight dropout applied to the trunk CastedLinear layers
    # (Attention.qkv_proj/o_proj and SwiGLU.gate_up_proj/down_proj). 0.0 = off.
    # The mask is resampled per training batch by the train loop.
    weight_dropout: float = 0.0

# ---- Consolidated upstream component 3 ----

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class SpecNormalizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.skip_power_iter = (self.in_features == 1 or self.out_features == 1)

        # 1. Weights - kept in float32 for spectral norm precision
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))

        # 2. Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.scale = nn.Parameter(torch.tensor(torch.linalg.matrix_norm(self.weight, ord=2).item()), requires_grad=True)
        self.scale._no_weight_decay = True

        self.register_buffer("u", torch.ones(out_features))
        self.register_buffer("v", torch.ones(in_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.skip_power_iter:
            # exact spectral norm for rank-1 matrices
            sigma = torch.norm(self.weight, p=2.0)
        else:
            u_init = self.u.detach().clone()
            v_init = self.v.detach().clone()
            # power iterations to estimate spectral norm
            sigma, u, v = self._power_iterations(A=self.weight, u_init=u_init, v_init=v_init, num_iterations=50)

            # store the results for the next forward pass
            with torch.no_grad():
                self.u = u.detach()
                self.v = v.detach()

        weight_sn = (self.weight / sigma) * self.scale

        return F.linear(
            input,
            weight_sn.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )

    @staticmethod
    @torch.compile(fullgraph=True)
    def _power_iterations(A: torch.Tensor, u_init: torch.Tensor, v_init: torch.Tensor, num_iterations: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power iterations to estimate spectral norm
        """
        v = v_init
        u = u_init

        for _ in range(num_iterations):
            v = nn.functional.normalize(torch.mv(A.t(), u), dim=0)
            u = nn.functional.normalize(torch.mv(A, v), dim=0)

        sigma = torch.dot(u, torch.mv(A, v))
        return sigma, u, v


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 dropout: float = 0.0):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

        self.dropout = dropout
        self.mask = nn.Buffer(torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask if self.training else self.weight
        return F.linear(input, weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)

    def reset_mask(self):
        if not self.training or self.dropout == 0.0:
            self.mask.fill_(1.0)
        else:
            self.mask.bernoulli_(1 - self.dropout).div_(1 - self.dropout)

class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False,
                 softmax_temp: float = 1.0, use_spec_norm_linear: bool = False, weight_dropout: float = 0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.softmax_temp = softmax_temp

        proj_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        # added feat: spectral normalization
        if use_spec_norm_linear:
            self.qkv_proj = SpecNormalizedLinear(self.hidden_size, proj_dim, bias=False)
        else:
            self.qkv_proj = CastedLinear(self.hidden_size, proj_dim, bias=False, dropout=weight_dropout)

        # added feat: spectral normalization
        if use_spec_norm_linear:
            self.o_proj = SpecNormalizedLinear(self.output_size, self.hidden_size, bias=False)
        else:
            self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False, dropout=weight_dropout)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, return_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal, scale=1.0 / (math.sqrt(self.head_dim) * self.softmax_temp))
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse: bool = False, use_spec_norm_linear: bool = False):
        super().__init__()

        if use_spec_norm_linear:
            self.linear = SpecNormalizedLinear(hidden_size, hidden_size, bias=False)
        else:
            self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, use_spec_norm_linear: bool = False, weight_dropout: float = 0.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        if use_spec_norm_linear:
            self.gate_up_proj = SpecNormalizedLinear(hidden_size, inter * 2, bias=False)
            self.down_proj    = SpecNormalizedLinear(inter, hidden_size, bias=False)
        else:
            self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False, dropout=weight_dropout)
            self.down_proj    = CastedLinear(inter, hidden_size, bias=False, dropout=weight_dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

# ---- Consolidated upstream component 4 ----

CosSin = Tuple[torch.Tensor, torch.Tensor]

'''
    modified some of the TRM blocks to use in the FP variant
    main modifications:
        - added conv1d over qkv and swiglu; absolutely necessary for FP stability
        - added QK normalization; helpful to stability
        - added weight norm to all linear layers; helpful to stability
'''


class FixedPointTransformerBlock(nn.Module):
    def __init__(self, config: ReasoningModelConfig) -> None:
        super().__init__()

        self.config = config
        weight_dropout = getattr(config, "weight_dropout", 0.0)
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
                weight_dropout=weight_dropout,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal,
                softmax_temp=config.softmax_temp,
                weight_dropout=weight_dropout,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            weight_dropout=weight_dropout,
        )
        self.norm_eps = config.rms_norm_eps
        self.norm_type = config.norm_type

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, alpha_1: torch.Tensor = None, beta_1: torch.Tensor = None) -> torch.Tensor:
        # B, L, D = hidden_states.shape

        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(rms_norm(hidden_states, variance_epsilon=self.norm_eps) if (self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm') else hidden_states)

            if self.norm_type == 'peri-norm':
                out = rms_norm(out, variance_epsilon=self.norm_eps)

            hidden_states = alpha_1.transpose(1, 2) * hidden_states + beta_1.transpose(1, 2) * out

            if self.norm_type == 'post-norm':
                hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            out = self.self_attn(
                cos_sin=cos_sin,
                hidden_states=rms_norm(hidden_states, variance_epsilon=self.norm_eps) if (self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm') else hidden_states,
            )

            if self.norm_type == 'peri-norm':
                out = rms_norm(out, variance_epsilon=self.norm_eps)

            hidden_states = alpha_1 * hidden_states + beta_1 * out

            if self.norm_type == 'post-norm':
                hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

        # Fully Connected
        out = self.mlp(rms_norm(hidden_states, variance_epsilon=self.norm_eps) if self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm' else hidden_states)

        if self.norm_type == 'peri-norm':
            out = rms_norm(out, variance_epsilon=self.norm_eps)

        hidden_states = alpha_1 * hidden_states + beta_1 * out

        if self.norm_type == 'post-norm':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

        return hidden_states


class FixedPointTransformer(nn.Module):
    def __init__(
        self,
        config: dict,
        n_layers: int
    ) -> None:
        super().__init__()

        self.config = config
        self.layers = torch.nn.ModuleList([FixedPointTransformerBlock(config) for _ in range(n_layers)])

        hidden_size = self.config.hidden_size
        # conv_params
        conv_kernel_size = self.config.conv_kernel_size
        conv_bias = self.config.conv_bias
        # residual scale params
        residual_scale = self.config.residual_scale
        self.outer_only = self.config.outer_only
        alpha_1_init = self.config.alpha_1_init
        alpha_2_init = self.config.alpha_2_init
        norm_placement = self.config.norm_placement
        normalize_input_injection = self.config.normalize_input_injection

        self.conv_type = self.config.conv_type

        if self.conv_type == 'conv1d':
            self.conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1,
                groups=hidden_size,
                bias=conv_bias,
            )
        elif self.conv_type == 'conv2d':
            if conv_kernel_size % 2 == 0:
                raise ValueError(f"conv_kernel_size must be odd for conv2d (got {conv_kernel_size})")
            self.conv = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=hidden_size,
                bias=conv_bias,
            )
        else:
            pass

        if residual_scale in ("none", "None"):
            residual_scale = None

        valid_residual_scale = {None, "fixed", "input-independent", "input-dependent"}
        if residual_scale not in valid_residual_scale:
            raise ValueError(f"Unknown residual_scale: {residual_scale}")

        valid_norm_placement = {"input", "output", "none"}
        if norm_placement not in valid_norm_placement:
            raise ValueError(f"Unknown norm_placement: {norm_placement}")

        if not (0.0 < alpha_1_init < 1.0):
            raise ValueError("alpha_1_init must be in (0, 1)")
        if not (0.0 < alpha_2_init < 1.0):
            raise ValueError("alpha_2_init must be in (0, 1)")

        self.residual_scale = residual_scale
        self.norm_placement = norm_placement
        self.normalize_input_injection = normalize_input_injection

        if self.residual_scale is not None:
            alpha_1_init_logit = math.log(alpha_1_init / (1 - alpha_1_init))
            alpha_2_init_logit = math.log(alpha_2_init / (1 - alpha_2_init))
            if self.residual_scale == 'fixed':
                self.alpha_2_param = nn.Parameter(torch.tensor(alpha_2_init), requires_grad=False)
                if not self.outer_only:
                    self.alpha_1_param = nn.Parameter(torch.tensor(alpha_1_init), requires_grad=False)

            elif self.residual_scale == 'input-independent':
                self.alpha_2_param = nn.Parameter(alpha_2_init_logit * torch.ones(hidden_size), requires_grad=True)
                self.alpha_2_param._no_weight_decay = True
                if not self.outer_only:
                    self.alpha_1_param = nn.Parameter(alpha_1_init_logit * torch.ones(hidden_size), requires_grad=True)
                    self.alpha_1_param._no_weight_decay = True

            elif self.residual_scale == 'input-dependent':
                if self.config.use_spec_norm_linear:
                    self.alpha_2_layer = SpecNormalizedLinear(hidden_size, hidden_size, bias=True)
                    if not self.outer_only:
                        self.alpha_1_layer = SpecNormalizedLinear(hidden_size, hidden_size, bias=True)

                else:
                    self.alpha_2_layer = CastedLinear(hidden_size, hidden_size, bias=True)
                    if not self.outer_only:
                        self.alpha_1_layer = CastedLinear(hidden_size, hidden_size, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.normalize_input_injection:
            input_injection = rms_norm(input_injection, variance_epsilon=self.config.rms_norm_eps)

        puzzle_emb_len = kwargs.pop('puzzle_emb_len', 0)

        if self.conv_type != 'none':
            batch_size, total_seq_len, hidden_size = hidden_states.shape
            puzzle_part = hidden_states[:, :puzzle_emb_len]
            grid_part = hidden_states[:, puzzle_emb_len:]

            grid_len = total_seq_len - puzzle_emb_len

            if self.conv_type == 'conv1d':
                grid_part = self.conv(grid_part.transpose(1, 2)).transpose(1, 2)[:, :grid_part.shape[1], :].contiguous()
            elif self.conv_type == 'conv2d':
                hw = int(math.sqrt(grid_len))
                assert hw * hw == grid_len, f"grid_len {grid_len} is not a perfect square"

                grid_part = grid_part.reshape(batch_size, hw, hw, hidden_size).permute(0, 3, 1, 2).contiguous()
                grid_part = self.conv(grid_part).permute(0, 2, 3, 1).reshape(batch_size, grid_len, hidden_size).contiguous()
            else:
                raise ValueError("Unknown convolution type.")

            hidden_states = torch.cat([puzzle_part, grid_part], dim=1).contiguous()
        else:
            pass

        if self.residual_scale is not None:
            if self.residual_scale == 'fixed':
                alpha_2 = self.alpha_2_param
                if not self.outer_only:
                    alpha_1 = self.alpha_1_param
            elif self.residual_scale == 'input-independent':
                alpha_2 = torch.sigmoid(self.alpha_2_param).to(dtype=hidden_states.dtype).reshape(1, 1, -1)
                if not self.outer_only:
                    alpha_1 = torch.sigmoid(self.alpha_1_param).to(dtype=hidden_states.dtype).reshape(1, 1, -1)
            elif self.residual_scale == 'input-dependent':
                alpha_2 = torch.sigmoid(self.alpha_2_layer(hidden_states)).to(dtype=hidden_states.dtype)
                if not self.outer_only:
                    alpha_1 = torch.sigmoid(self.alpha_1_layer(hidden_states)).to(dtype=hidden_states.dtype)

            # weight tying for beta_1, beta_2
            # based on asymptotic analysis
            if self.outer_only:
                one = hidden_states.new_ones(1, 1, 1)
                beta_2 = 1 - alpha_2
                alpha_1, beta_1 = one, one
            else:
                beta_2 = 1 - alpha_2 * alpha_1.pow(2 * len(self.layers))
                beta_1 = beta_2 * (1 - alpha_1) / (1 - alpha_1.pow(2 * len(self.layers)) + 1e-5)
        else:
            one = hidden_states.new_ones(1, 1, 1)
            alpha_1, alpha_2, beta_1, beta_2 = one, one, one, one

        if self.norm_placement == 'input':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.config.rms_norm_eps)

        hidden_states = alpha_2 * hidden_states + beta_2 * input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                alpha_1=alpha_1,
                beta_1=beta_1,
                **kwargs,
            )

        if self.norm_placement == 'output':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.config.rms_norm_eps)

        return hidden_states

# ---- Consolidated upstream component 5 ----

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training mode, fill puzzle embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,

        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None

            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False

            assert local_ids is not None
            assert weights is not None

            # Apply SignSGD
            # Adam ≈ SignSGD if gradient is very sparse
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,

                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"]
                )


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,

    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape

    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)

        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated slices back
    weights[grad_ids] = p

# ---- Consolidated upstream component 6 ----

def shift(input:torch.Tensor, shift, dim=0, fillval=0):
    # torch.roll without the copy of the wrap-around section
    size = input.size(dim)
    fill = torch.full_like(input.narrow(dim, 0, abs(shift)), fillval)
    if shift > 0:
        output = torch.cat([fill, input.narrow(dim, 0, size-shift)], dim=dim)
    if shift < 0:
        output = torch.cat([input.narrow(dim, -shift, size+shift), fill], dim=dim)
    return output

class FixedPointOptimizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.stepsize = config.stepsize
        self.stepsize_decay_train = config.stepsize_decay_train
        self.stepsize_decay_eval = config.stepsize_decay_eval
        self.decay_patience = config.decay_patience
        self.eps = config.eps
        self.outlier_quantile = config.outlier_quantile
        self.max_iter = config.max_iter
        self.init_std = config.init_std
        self.additive_noise_std = config.additive_noise_std
        self.fp_thresh = config.fp_thresh

        self.fixed_init = config.fixed_init
        if self.fixed_init:
            fwd_dtype = getattr(torch, config.forward_dtype)
            self.init_vec = nn.Buffer(
                trunc_normal_init_(torch.empty(config.hidden_size, dtype=fwd_dtype), std=self.init_std),
                persistent=True,
            )

    def detach_state(self, state: dict):
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach()
        return state

    def reset(self, reset_flag: torch.Tensor, shape: tuple, dtype: torch.dtype,
              device: torch.device, state: dict, reset_metadata: bool = False):
        batch_size, seq_len, hidden_size = shape[0], shape[1], shape[2]
        reset_flag_1d = reset_flag.view(-1)
        reset_flag_3d = reset_flag_1d.view(-1, 1, 1)

        if self.fixed_init:
            y = self.init_vec.to(dtype=dtype, device=device).expand(batch_size, seq_len, hidden_size).contiguous()
        else:
            y = trunc_normal_init_(torch.empty(batch_size, seq_len, hidden_size, dtype=dtype, device=device), std=self.init_std)
        residues = torch.inf * torch.ones(batch_size).to(device)
        stepsize = (self.stepsize * torch.ones(batch_size, 1, 1, dtype=dtype, device=device))
        patience = self.decay_patience * torch.ones(batch_size).to(device)
        iter_idx = torch.zeros(batch_size, dtype=torch.int32, device=device)
        best_residues = torch.inf * torch.ones(batch_size).to(device)

        if state is None:
            state = dict(y=y.contiguous(),
                         residues=residues,
                         stepsize=stepsize,
                         patience=patience,
                         iter_idx=iter_idx,
                         best_residues=best_residues)
        else:
            state = dict(y=torch.where(reset_flag_3d, y.contiguous(), state['y']),
                         residues=residues if reset_metadata else torch.where(reset_flag_1d, residues, state['residues']),
                         stepsize=stepsize if reset_metadata else torch.where(reset_flag_3d, stepsize, state['stepsize']),
                         patience=patience if reset_metadata else torch.where(reset_flag_1d, patience, state['patience']),
                         iter_idx=iter_idx if reset_metadata else torch.where(reset_flag_1d, iter_idx, state['iter_idx']),
                         best_residues=best_residues if reset_metadata else torch.where(reset_flag_1d, best_residues, state['best_residues']))

        return state

    def step(self, state:Dict[str, torch.Tensor], y:torch.Tensor):
        state_dtype = state["y"].dtype
        if y.dtype != state_dtype:
            y = y.to(state_dtype)

        with torch.no_grad():
            residues = (state['y'].detach() - y.detach()).norm(p=torch.inf, dim=-1) / (y.detach().norm(p=torch.inf, dim=-1) + self.eps)
            residues = residues.max(dim=1)[0]

        stepsize = state['stepsize']
        if stepsize.dtype != state_dtype:
            stepsize = stepsize.to(state_dtype)
        state['y'] = y * stepsize + state['y'] * (1 - stepsize) + self.additive_noise_std * torch.randn_like(state['y'])

        improved = residues < state['best_residues']

        # update patience and lowest residue
        state['residues'] = residues
        state['best_residues'] = torch.where(improved, residues, state['best_residues'])
        state['patience'] = torch.where(improved, self.decay_patience, state['patience']-1)

        # update damping factor, reset patience
        adapt = (state['patience'] <= 0) & (state['residues'] >= self.fp_thresh)
        state['patience'] = torch.where(adapt, self.decay_patience, state['patience'])
        stepsize_dtype = state['stepsize'].dtype
        # Separate train/eval decay (selected by module mode set via model.train()/eval()).
        decay = self.stepsize_decay_train if self.training else self.stepsize_decay_eval
        state['stepsize'] = state['stepsize'] * torch.where(adapt, decay, 1).to(stepsize_dtype).reshape(-1, 1, 1)

        state['iter_idx'] = state['iter_idx'] + 1
        return state

    def cont(self, state: Dict[str, torch.Tensor], thresh: float):
        if int(state['iter_idx'].max().item()) == 0:
            return self.max_iter > 0

        q = 1 - self.outlier_quantile if self.training else 1.0
        return (
            (torch.quantile(state['residues'].float(), q=q) >= thresh)
            & (torch.quantile(state['iter_idx'].float(), q=q) < self.max_iter)
            & (torch.quantile(state['stepsize'].float(), q=q) > 1e-3)
        )

class VariationalDropout(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        self.dropout = dropout
        self._mask: torch.Tensor | None = None

    def sample_mask(self, x: torch.Tensor) -> None:
        if not self.training or self.dropout == 0.0:
            self._mask = None
            return
        keep_prob = 1.0 - self.dropout
        self._mask = torch.bernoulli(torch.full_like(x, keep_prob)) / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0 or self._mask is None:
            return x
        return x * self._mask


class VariationalDropToken1d(nn.Module):
    def __init__(self, dropout: float = 0.0, token_first: bool = True):
        super().__init__()
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        self.dropout = dropout
        self.token_first = token_first
        self._mask: torch.Tensor | None = None

    def sample_mask(self, x: torch.Tensor) -> None:
        if not self.training or self.dropout == 0.0:
            self._mask = None
            return
        keep_prob = 1.0 - self.dropout
        if self.token_first:
            B, L, _ = x.shape
            self._mask = torch.bernoulli(torch.full((B, L, 1), keep_prob, device=x.device, dtype=x.dtype)) / keep_prob
        else:
            B, _, L = x.shape
            self._mask = torch.bernoulli(torch.full((B, 1, L), keep_prob, device=x.device, dtype=x.dtype)) / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0 or self._mask is None:
            return x
        return x * self._mask

# ---- Consolidated upstream component 7 ----

class FPRMConfig(ReasoningModelConfig):
    # Halting fields are required on the base class but unused by most FP archs;
    # default them so existing FP yamls don't need to set them.
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    halting_mechanism: Literal["act", "fixed_point", "fixed_iterations"] = "fixed_point"

    # Fixed-point iteration controls
    max_iter: int
    # Eval-time max_iter cap. None = fall back to max_iter
    max_iter_eval: Optional[int] = None

    # Variational dropout on the L_level output (multiplicative mask). Same
    # mask is reused for all FP iterations on a given sample (resampled when
    # the sample is halted). 0.0 = off.
    variational_dropout: float = 0.0

    # Jacobian-norm regularizer (Hutchinson estimator).
    # jacobian_reg selects the JVP method:
    #   "none"  - regularizer disabled (no extra L_level forwards, no outputs["jacobian_loss"])
    #   "fda"   - central-difference finite approximation (2 extra forwards per sample)
    #   "exact" - autograd VJP with create_graph=True (1 forward + 1 backward per sample;
    #             requires MATH SDPA backend for double-backward)
    # n_jacobian_samples is the number of Hutchinson samples; jacobian_eps is the
    # central-difference step (only used by "fda").
    jacobian_reg: Literal["none", "fda", "exact"] = "none"
    n_jacobian_samples: int = 0
    jacobian_eps: float = 1.0e-3

    # Fixed-point solver
    fp_thresh: float = 0.1
    outlier_quantile: float = 0.25
    stepsize: float = 1.0
    # Separate FP stepsize-decay rates for training vs eval. The optimizer
    # picks by self.training, so training-time eval (and inference) can use a
    # different (typically slower) decay than training without a separate
    # script — the eval curves logged to W&B reflect stepsize_decay_eval.
    stepsize_decay_train: float = 0.9
    stepsize_decay_eval: float = 0.99
    decay_patience: int = 5
    eps: float = 1e-8
    # Std of the truncated-normal init for the FP iterate y at carry reset.
    init_std: float = 1.0
    # Std of additive Gaussian noise injected at every FP iteration update.
    # 0.0 = off.
    additive_noise_std: float = 0.0
    # If True, use a persistent (hidden_size,) buffer (random at construction,
    # frozen thereafter) broadcast over (batch, seq, hidden) at reset — matching
    # TRM's H_init/L_init scheme. If False (default), draw fresh i.i.d. random
    # of shape (batch, seq, hidden) at every reset.
    fixed_init: bool = False
    gamma_alpha: float = 4.0
    gamma_scale: float = 50 / 3
    max_iter_dist: str = "gamma"
    expon_scale: float = 50.0 / math.log(2.0)

# ---- Consolidated upstream component 8 ----

IGNORE_LABEL_ID = -100


@dataclass
class FixedPointReasoningModel_ACTV1InnerCarry:
    z_L_state: dict
    dropout_mask: torch.Tensor

@dataclass
class FixedPointReasoningModel_ACTV1Carry:
    inner_carry: FixedPointReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class FixedPointReasoningModel_Inner(nn.Module):
    def __init__(self, config: FPRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = FixedPointTransformer(self.config, self.config.L_layers)

        self.L_optimizer = FixedPointOptimizer(self.config)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return FixedPointReasoningModel_ACTV1InnerCarry(
            z_L_state = None,
            dropout_mask = None,
        )

    def reset_carry(self,
                    reset_flag: torch.Tensor,
                    batch: torch.Tensor,
                    carry: FixedPointReasoningModel_ACTV1InnerCarry):
        shape = (batch.shape[0], batch.shape[1] + self.puzzle_emb_len, self.config.hidden_size)
        device = batch.device
        dtype = self.forward_dtype

        if self.training:
            dropout_mask = torch.empty(*shape, device=device, dtype=dtype).bernoulli_(p=1 - self.config.variational_dropout).div_(1 - self.config.variational_dropout)
            if carry.dropout_mask is not None:
                dropout_mask = torch.where(reset_flag.view(-1, 1, 1), dropout_mask, carry.dropout_mask)
        else:
            dropout_mask = torch.ones(*shape, device=device, dtype=dtype)

        return FixedPointReasoningModel_ACTV1InnerCarry(
            z_L_state = self.L_optimizer.reset(reset_flag, shape, dtype, device, carry.z_L_state),
            dropout_mask=dropout_mask,
        )

    # Disable torch.compile for this method: the 'exact' branch backprops through
    # autograd.grad(..., create_graph=True), and torch.compile's aot_autograd
    # backend does not support double-backward. Eager-mode autograd does.
    @torch._dynamo.disable
    def _jacobian_reg(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                      dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        # Eval path discards the regularizer loss; skip the work. Required for
        # 'exact' too, where autograd.grad would fail under eval's no_grad.
        if not self.training or self.config.jacobian_reg == 'none':
            return state["y"].new_zeros(())

        estimate = 0

        from torch.nn.attention import SDPBackend, sdpa_kernel
        _sdpa_ctx = sdpa_kernel(SDPBackend.MATH)

        for n in range(self.config.n_jacobian_samples):
            v = (torch.randn_like(state["y"]) / math.sqrt(state["y"].shape[-1])).detach()
            z_in = state["y"].detach().requires_grad_(True)

            if self.config.jacobian_reg == 'fda':
                z_p = dropout_mask * self.L_level(z_in + self.config.jacobian_eps * v, input_embeddings, **seq_info)
                z_m = dropout_mask * self.L_level(z_in - self.config.jacobian_eps * v, input_embeddings, **seq_info)

                jvp = (z_p - z_m) / (2 * self.config.jacobian_eps)

                estimate += jvp.pow(2).mean() / self.config.n_jacobian_samples

            elif self.config.jacobian_reg == 'exact':
                with torch.enable_grad(), _sdpa_ctx:
                    f_z = dropout_mask * self.L_level(z_in, input_embeddings, **seq_info)
                    s = (v * f_z).sum()
                    # create_graph=True so the loss can backprop through Jt_v;
                    # retain_graph defaults to create_graph here (we need it).
                    (jvp,) = torch.autograd.grad(s, z_in, create_graph=True)
                estimate += jvp.pow(2).mean() / self.config.n_jacobian_samples

            else:
                raise ValueError(f"Unknown jacobian_reg: {self.config.jacobian_reg!r}")

        return estimate

    def _z_step(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        z_new = dropout_mask * self.L_level(state["y"], input_embeddings, **seq_info)
        return self.L_optimizer.step(state, z_new)

    def forward(self,
                carry: FixedPointReasoningModel_ACTV1InnerCarry,
                batch: Dict[str, torch.Tensor],
                force_grad: bool,
                n_steps: int) -> Tuple[
        Tuple[FixedPointReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Slice rotary cache to the actual input length so the same model
        # can be evaluated at sequence lengths shorter than config.seq_len
        # (length-generalisation for state tracking).
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=self.puzzle_emb_len)
        z_state, dropout_mask = carry.z_L_state, carry.dropout_mask

        with torch.set_grad_enabled(force_grad):
            for _ in range(n_steps):
                z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info)

        output = self.lm_head(z_state['y'])[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_state['y'][:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        jacobian_loss = self._jacobian_reg(z_state, input_embeddings, dropout_mask, seq_info)
        new_carry = FixedPointReasoningModel_ACTV1InnerCarry(z_L_state=self.L_optimizer.detach_state(z_state),
                                                                         dropout_mask=carry.dropout_mask)  # New carry no grad
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1], jacobian_loss)


class FixedPointReasoningModel_ACTV1(nn.Module):
    """Single-state FPTRM wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FPRMConfig(**config_dict)
        self.inner = FixedPointReasoningModel_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return FixedPointReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def set_num_iters(self):
        if self.training:
            if self.config.max_iter_dist == 'gamma':
                sampled_max_iter = gamma.rvs(a=self.config.gamma_alpha, scale=self.config.gamma_scale)
            elif self.config.max_iter_dist == 'expon':
                sampled_max_iter = expon.rvs(scale=self.config.expon_scale)
            elif self.config.max_iter_dist == 'det':
                sampled_max_iter = self.config.max_iter

            if self.config.max_iter_dist == 'det':
                self.max_iter = max(0, int(sampled_max_iter))
            else:
                self.max_iter = max(1, int(sampled_max_iter))
        else:
            self.max_iter = self.config.max_iter_eval if self.config.max_iter_eval is not None else self.config.max_iter

    def forward(
        self,
        carry: FixedPointReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ):
        # Update data, carry (removing halted sequences)
        # Handled inside the optimizer
        new_inner_carry = self.inner.reset_carry(carry.halted, batch['inputs'], carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward-backward inner model
        n_steps = self.config.n_backwards_L if self.training else 1
        new_inner_carry, logits, (q_halt_logits, q_continue_logits, jacobian_loss) = self.inner(new_inner_carry, new_current_data,
                                                                                                force_grad=self.training, n_steps=n_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        if self.training and self.config.jacobian_reg != 'none' and self.config.n_jacobian_samples > 0:
            outputs["jacobian_loss"] = jacobian_loss

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1

            if self.config.halting_mechanism == 'act':
                is_last_step = new_steps >= self.config.halt_max_steps
            else:
                is_last_step = new_steps >= self.max_iter

            halted = is_last_step

            # if testing, use fixed-points
            if not self.training:
                # during inference we only halt for the entire sequence
                halted = halted | (new_inner_carry.z_L_state['residues'].max() < self.config.fp_thresh) \
                                | (new_inner_carry.z_L_state['stepsize'].max() < 1e-3)

            # if training, and ACT is enabled
            cap = self.config.halt_max_steps if self.config.halting_mechanism == 'act' else self.max_iter
            if self.training and (cap > 1):

                if self.config.halting_mechanism == 'act':
                    if self.config.no_ACT_continue:
                        halted = halted | (q_halt_logits > 0)
                    else:
                        halted = halted | (q_halt_logits > q_continue_logits)

                    # Exploration
                    min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                elif self.config.halting_mechanism == 'fixed_point':
                    # Exploration is implemented by self.max_iter if we choose to use it
                    halted = halted | (new_inner_carry.z_L_state['residues'] < self.config.fp_thresh) \
                                    | (new_inner_carry.z_L_state['stepsize'].view(-1) < 1e-3)

                elif self.config.halting_mechanism == 'fixed_iterations':
                    pass

                else:
                    raise ValueError("FPRM only accepts ACT, Fixed_point, and Fixed_iterations as its halting mechanism.")

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits, _) = self.inner(new_inner_carry, new_current_data, force_grad=False, n_steps=1)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return FixedPointReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

# ---- Consolidated upstream component 9 ----

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, q_loss_coeff: float = 0.5,
                 deep_supervision: bool = True, jacobian_reg_lambda: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.deep_supervision = deep_supervision
        self.q_loss_coeff = q_loss_coeff
        self.jacobian_reg_lambda = jacobian_reg_lambda

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def set_num_iters(self, *args, **kwargs):
        if hasattr(self.model, "set_num_iters"):
            return self.model.set_num_iters(*args, **kwargs)  # type: ignore
        return None

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "sequence_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        if self.deep_supervision:
            masked_loss = lm_loss
        else:
            lm_mask = mask & new_carry.halted.unsqueeze(-1)
            lm_loss_counts = lm_mask.sum(-1)
            lm_loss_divisor = lm_loss_counts.clamp_min(1).unsqueeze(-1)
            masked_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=lm_mask) / lm_loss_divisor).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        jacobian_loss = 0
        if "jacobian_loss" in outputs:
            jacobian_loss = outputs["jacobian_loss"]
            metrics["jacobian_loss"] = outputs["jacobian_loss"].detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        masked_full_loss = masked_loss + self.q_loss_coeff * (q_halt_loss + q_continue_loss) + self.jacobian_reg_lambda * jacobian_loss
        full_loss = lm_loss + self.q_loss_coeff * (q_halt_loss + q_continue_loss) + self.jacobian_reg_lambda * jacobian_loss

        return new_carry, masked_full_loss, full_loss, metrics, detached_outputs, new_carry.halted.all()
FixedPointReasoningModel = FixedPointReasoningModel_ACTV1
CastedSparseEmbeddingSignSGDDistributed = CastedSparseEmbeddingSignSGD_Distributed


def normalize_fprm_state_dict(
    state_dict: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """Remove the documented compile wrapper without guessing key mappings."""

    return normalize_state_dict_keys(state_dict, prefixes=("_orig_mod.",))


def _normalize_fprm_model_state_dict(
    state_dict: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """Normalize released loss-head keys for the unwrapped model."""

    return normalize_state_dict_keys(
        state_dict,
        prefixes=("_orig_mod.model.", "_orig_mod.", "model."),
    )


class FPRMAdapter:
    """Translate FPRM-native carries and losses to the common RRM contract."""

    name = "fprm"

    def build_model(
        self, task: str, preset: str, metadata: Mapping[str, Any]
    ) -> FixedPointReasoningModel:
        if task not in ("maze-hard", "sudoku-extreme"):
            raise ValueError(f"FPRM does not support task {task!r}")
        arch = dict(metadata.get("arch", metadata))
        for key in ("batch_size", "seq_len", "num_puzzle_identifiers", "vocab_size"):
            if key in metadata:
                arch[key] = metadata[key]
        model = FixedPointReasoningModel(arch)
        loss = dict(arch.get("loss", metadata.get("loss", {})))
        model._rrm_q_loss_coeff = float(loss.get("q_loss_coeff", 0.5))
        model._rrm_jacobian_reg_lambda = float(loss.get("jacobian_reg_lambda", 0.0))
        return model

    def load_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> Mapping[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("FPRM checkpoint must contain a state mapping")
        if payload.get("format") == "RRM_CHECKPOINT_V1":
            state = payload.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError("RRM checkpoint is missing model_state")
        elif "model" in payload and isinstance(payload["model"], Mapping):
            state = payload["model"]
        else:
            state = payload
        model.load_state_dict(_normalize_fprm_model_state_dict(state), strict=True)
        return payload

    def training_forward(
        self,
        model: nn.Module,
        state: object,
        batch: Mapping[str, Tensor],
    ) -> TrainingOutput:
        if not isinstance(model, FixedPointReasoningModel_ACTV1):
            raise TypeError("FPRMAdapter requires FixedPointReasoningModel")
        model.set_num_iters()
        for module in model.modules():
            if hasattr(module, "reset_mask"):
                module.reset_mask()
        new_carry, native = model(carry=state, batch=dict(batch))
        labels = new_carry.current_data["labels"]
        valid = labels != IGNORE_LABEL_ID
        counts = valid.sum(dim=-1).clamp_min(1)
        token_loss = stablemax_cross_entropy(
            native["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid
        )
        task_loss = token_loss.sum(dim=-1) / counts
        predictions = native["logits"].argmax(dim=-1)
        sequence_correct = ((predictions == labels) | ~valid).all(dim=-1)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            native["q_halt_logits"],
            sequence_correct.to(native["q_halt_logits"].dtype),
            reduction="sum",
        )
        q_continue_loss = native["q_halt_logits"].new_zeros(())
        if "target_q_continue" in native:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                native["q_continue_logits"],
                native["target_q_continue"],
                reduction="sum",
            )
        jacobian_loss = native.get(
            "jacobian_loss", native["q_halt_logits"].new_zeros(())
        )
        auxiliary_loss = (
            float(getattr(model, "_rrm_q_loss_coeff", 0.5))
            * (q_halt_loss + q_continue_loss)
            + float(getattr(model, "_rrm_jacobian_reg_lambda", 0.0))
            * jacobian_loss
        )
        return TrainingOutput(
            state=new_carry,
            per_example_task_loss=task_loss,
            auxiliary_loss=auxiliary_loss,
            predictions=predictions,
            selection_score=native["q_halt_logits"],
            metrics={
                "sequence_accuracy": sequence_correct.float().mean().detach(),
                "q_halt_loss": q_halt_loss.detach(),
            },
        )

    @torch.no_grad()
    def evaluation_forward(
        self, model: nn.Module, batch: Mapping[str, Tensor]
    ) -> EvaluationOutput:
        if not isinstance(model, FixedPointReasoningModel_ACTV1):
            raise TypeError("FPRMAdapter requires FixedPointReasoningModel")
        was_training = model.training
        model.eval()
        model.set_num_iters()
        native_batch = dict(batch)
        carry = model.initial_carry(native_batch)
        batch_size, sequence_length = native_batch["inputs"].shape
        predictions = torch.zeros(
            batch_size, sequence_length, dtype=torch.long, device=native_batch["inputs"].device
        )
        selection_score = torch.zeros(
            batch_size, dtype=torch.float32, device=native_batch["inputs"].device
        )
        steps = torch.zeros(
            batch_size, dtype=torch.int32, device=native_batch["inputs"].device
        )
        done = torch.zeros(
            batch_size, dtype=torch.bool, device=native_batch["inputs"].device
        )
        maximum = int(model.max_iter)
        for _ in range(maximum + 1):
            carry, native = model(carry=carry, batch=native_batch)
            newly_done = carry.halted & ~done
            candidate_predictions = native["logits"].argmax(dim=-1)
            predictions = torch.where(
                newly_done.unsqueeze(-1), candidate_predictions, predictions
            )
            selection_score = torch.where(
                newly_done, native["q_halt_logits"].float(), selection_score
            )
            steps = torch.where(newly_done, carry.steps, steps)
            done |= newly_done
            if bool(done.all()):
                break
        if was_training:
            model.train()
        if not bool(done.all()):
            raise RuntimeError("FPRM did not halt every sequence within max_iter")
        return EvaluationOutput(
            predictions=predictions,
            selection_score=selection_score,
            halted=done,
            steps=steps,
        )

    def build_optimizers(
        self, model: nn.Module, preset: Mapping[str, Any]
    ) -> tuple[Optimizer, Optimizer | None]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(
                parameter
            )
        learning_rate = float(preset.get("learning_rate", preset.get("lr", 1e-4)))
        weight_decay = float(preset.get("weight_decay", 1e-4))
        dense = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(float(preset.get("beta1", 0.9)), float(preset.get("beta2", 0.999))),
        )
        sparse: Optimizer | None = None
        base_model = model.model if isinstance(model, ACTLossHead) else model
        if getattr(base_model.config, "puzzle_emb_ndim", 0) > 0:
            sparse = CastedSparseEmbeddingSignSGD_Distributed(
                base_model.puzzle_emb.buffers(),
                world_size=int(preset.get("world_size", 1)),
                lr=float(preset.get("puzzle_embedding_learning_rate", 0.01)),
                weight_decay=float(preset.get("puzzle_embedding_weight_decay", 1.0)),
            )
        return dense, sparse


ADAPTER = FPRMAdapter()
