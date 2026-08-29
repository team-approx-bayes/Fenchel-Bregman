"""PTRM/TRM baseline and sampling utilities in one public module.

The model implementation below is mechanically consolidated from
TinyRecursiveModels at commit c01103738605ba39d1430519b1ee0c62f4c707f8:
https://github.com/SamsungSAILMontreal/TinyRecursiveModels

MIT License

Copyright (c) 2025. Samsung Electronics Co., Ltd. All Rights Reserved.

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

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import einops
from pydantic import BaseModel
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention
from torch.optim.optimizer import Optimizer, ParamsT

from rrm.utils import EvaluationOutput, TrainingOutput, normalize_state_dict_keys, sha256_file


TRM_UPSTREAM_COMMIT = "c01103738605ba39d1430519b1ee0c62f4c707f8"
TRM_UPSTREAM_URL = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels"

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


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


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
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

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

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

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
    cellwise_q_loss: bool = False

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
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
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

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
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        q_cell_logits: torch.Tensor | None = None
        q_select_logits: torch.Tensor | None = None
        if self.config.cellwise_q_loss:
            q_cell_logits = self.q_head(z_H[:, self.puzzle_emb_len:])[..., 0].to(
                torch.float32
            )
            q_select_logits = smooth_min_cell_logits(q_cell_logits)
        return new_carry, output, (
            q_logits[..., 0],
            q_logits[..., 1],
            q_cell_logits,
            q_select_logits,
        )


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, q_outputs = self.inner(
            new_inner_carry, new_current_data
        )
        q_halt_logits, q_continue_logits, q_cell_logits, q_select_logits = q_outputs

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        if q_cell_logits is not None and q_select_logits is not None:
            outputs["q_cell_logits"] = q_cell_logits
            outputs["q_select_logits"] = q_select_logits

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, next_q_outputs = self.inner(
                        new_inner_carry, new_current_data
                    )
                    next_q_halt_logits, next_q_continue_logits = next_q_outputs[:2]
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

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


def aggregate_token_loss(
    token_losses: Tensor,
    valid_mask: Tensor,
    *,
    mode: str = "mean",
    smooth_delta: float = math.log(2.0),
) -> Tensor:
    """Reduce token losses to one per-example value for the FB objective."""

    if token_losses.ndim != 2 or valid_mask.shape != token_losses.shape:
        raise ValueError("token losses and valid mask must be matching matrices")
    counts = valid_mask.sum(dim=-1)
    if mode == "mean":
        return token_losses.masked_fill(~valid_mask, 0).sum(dim=-1) / counts.clamp_min(1)
    if mode != "smooth_weakest":
        raise ValueError(f"unsupported token-loss aggregation: {mode}")
    if not math.isfinite(float(smooth_delta)) or smooth_delta <= 0:
        raise ValueError("smooth_delta must be positive and finite")
    safe_counts = counts.clamp_min(1).to(token_losses.dtype)
    log_counts = safe_counts.log()
    temperature = float(smooth_delta) / log_counts.clamp_min(1e-12)
    scaled = token_losses / temperature.unsqueeze(-1)
    smooth = temperature * (
        torch.logsumexp(scaled.masked_fill(~valid_mask, float("-inf")), dim=-1)
        - log_counts
    )
    single = token_losses.masked_fill(~valid_mask, 0).sum(dim=-1)
    return torch.where(counts > 1, smooth, torch.where(counts > 0, single, 0))


def smooth_min_cell_logits(cell_logits: Tensor) -> Tensor:
    """Aggregate cell correctness logits without changing the Q-head."""

    if cell_logits.ndim != 2 or cell_logits.shape[1] < 1:
        raise ValueError("cell logits must be a non-empty matrix")
    values = cell_logits.to(torch.float32)
    return -(
        torch.logsumexp(-values, dim=1) - math.log(values.shape[1])
    )


@dataclass(frozen=True)
class CellwiseQLossTerms:
    sequence_exact: Tensor
    prefix_loss: Tensor
    cell_loss: Tensor
    aggregate_loss: Tensor
    halt_loss: Tensor
    aggregate_logits: Tensor


def cellwise_q_loss_terms(
    *,
    prefix_logits: Tensor,
    cell_logits: Tensor,
    predictions: Tensor,
    labels: Tensor,
    valid_mask: Tensor,
) -> CellwiseQLossTerms:
    """Average whole-answer, balanced cell, and aggregate Q supervision."""

    if predictions.shape != labels.shape or valid_mask.shape != labels.shape:
        raise ValueError("predictions, labels, and valid mask must have matching shapes")
    if cell_logits.shape != labels.shape or prefix_logits.shape != labels.shape[:1]:
        raise ValueError("Q logits do not match the prediction batch")
    correct = predictions.eq(labels) & valid_mask
    wrong = ~predictions.eq(labels) & valid_mask
    sequence_exact = (correct | ~valid_mask).all(dim=-1)
    targets = sequence_exact.to(torch.float32)
    prefix_loss = F.binary_cross_entropy_with_logits(
        prefix_logits.to(torch.float32), targets, reduction="none"
    )
    values = cell_logits.to(torch.float32)
    correct_loss = F.softplus(-values)
    wrong_loss = F.softplus(values)
    correct_count = correct.sum(dim=-1)
    wrong_count = wrong.sum(dim=-1)
    correct_mean = (correct_loss * correct).sum(dim=-1) / correct_count.clamp_min(1)
    wrong_mean = (wrong_loss * wrong).sum(dim=-1) / wrong_count.clamp_min(1)
    has_correct = correct_count > 0
    has_wrong = wrong_count > 0
    cell_loss = torch.where(
        has_correct & has_wrong,
        (correct_mean + wrong_mean) / 2.0,
        torch.where(has_correct, correct_mean, wrong_mean),
    )
    aggregate_logits = smooth_min_cell_logits(values)
    aggregate_loss = F.binary_cross_entropy_with_logits(
        aggregate_logits, targets, reduction="none"
    )
    return CellwiseQLossTerms(
        sequence_exact=sequence_exact,
        prefix_loss=prefix_loss,
        cell_loss=cell_loss,
        aggregate_loss=aggregate_loss,
        halt_loss=(prefix_loss + cell_loss + aggregate_loss) / 3.0,
        aggregate_logits=aggregate_logits,
    )


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

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
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
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
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

TinyRecursiveReasoningModel = TinyRecursiveReasoningModel_ACTV1
TinyRecursiveReasoningModelInnerCarry = TinyRecursiveReasoningModel_ACTV1InnerCarry
CastedSparseEmbeddingSignSGDDistributed = CastedSparseEmbeddingSignSGD_Distributed


def normalize_ptrm_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Normalize a released loss-head checkpoint without guessing key names."""

    return dict(normalize_state_dict_keys(state, prefixes=("_orig_mod.",)))


def _normalize_ptrm_base_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return dict(
        normalize_state_dict_keys(
            state,
            prefixes=("_orig_mod.model.", "_orig_mod.", "model."),
        )
    )


def select_first_max(scores: Tensor) -> Tensor:
    """Return the first maximum candidate for every row."""

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [batch, candidates], got {scores.shape}")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return torch.argmax(scores, dim=-1)


def add_latent_noise(
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    *,
    sigma: float,
    generator: torch.Generator | None = None,
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """Add posterior noise only to the low-level recurrent latent."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    noise = torch.randn(
        carry.z_L.shape,
        dtype=carry.z_L.dtype,
        device=carry.z_L.device,
        generator=generator,
    )
    return type(carry)(z_H=carry.z_H, z_L=carry.z_L + sigma * noise)


@contextmanager
def sampled_parameters(
    model: nn.Module,
    *,
    scale: float,
    generator: torch.Generator | None = None,
) -> Iterator[None]:
    """Temporarily sample independent Gaussian parameter perturbations."""

    if scale < 0:
        raise ValueError("scale must be non-negative")
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    ]
    originals = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter in parameters:
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                parameter.add_(noise, alpha=scale)
        yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)


def parameter_mean_abs(model: nn.Module) -> float:
    """Return the element-weighted mean absolute trainable parameter value."""

    absolute_sum = 0.0
    element_count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            absolute_sum += float(parameter.detach().float().abs().sum().item())
            element_count += parameter.numel()
    if element_count == 0:
        raise ValueError("model has no trainable parameters")
    return absolute_sum / element_count


class PTRMAdapter:
    """Translate TRM carries, losses, and posterior candidates to RRM outputs."""

    name = "ptrm"

    def build_model(
        self, task: str, preset: str, metadata: Mapping[str, Any]
    ) -> TinyRecursiveReasoningModel:
        if task not in ("maze-hard", "sudoku-extreme"):
            raise ValueError(f"PTRM does not support task {task!r}")
        arch = dict(metadata.get("arch", metadata))
        for key in ("batch_size", "seq_len", "num_puzzle_identifiers", "vocab_size"):
            if key in metadata:
                arch[key] = metadata[key]
        loss = dict(arch.get("loss", metadata.get("loss", {})))
        cellwise_q_loss = bool(loss.get("cellwise_q_loss", False))
        if cellwise_q_loss and task != "maze-hard":
            raise ValueError("cellwise Q loss is only frozen for Maze-Hard")
        arch["cellwise_q_loss"] = cellwise_q_loss
        model = TinyRecursiveReasoningModel(arch)
        model._rrm_q_loss_coeff = float(loss.get("q_loss_coeff", 0.5))
        model._rrm_success_loss_aggregation = str(
            loss.get("success_loss_aggregation", "mean")
        )
        model._rrm_smooth_weakest_delta = float(
            loss.get("smooth_weakest_delta", math.log(2.0))
        )
        if preset == "trm":
            candidate_count = 1
            latent_noise_sigma = 0.0
        else:
            candidate_count = int(metadata.get("candidate_count", 10))
            latent_noise_sigma = float(metadata.get("latent_noise_sigma", 0.3))
        inference_depth = int(
            metadata.get("inference_depth", arch.get("halt_max_steps", 1))
        )
        if candidate_count < 1 or inference_depth < 1:
            raise ValueError("candidate_count and inference_depth must be positive")
        if latent_noise_sigma < 0:
            raise ValueError("latent_noise_sigma must be non-negative")
        model._rrm_candidate_count = candidate_count
        model._rrm_latent_noise_sigma = latent_noise_sigma
        model._rrm_inference_depth = inference_depth
        return model

    def load_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> Mapping[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("PTRM checkpoint must contain a state mapping")
        if payload.get("format") == "RRM_CHECKPOINT_V1":
            state = payload.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError("RRM checkpoint is missing model_state")
        elif "model_state_dict" in payload and isinstance(
            payload["model_state_dict"], Mapping
        ):
            state = payload["model_state_dict"]
        elif "model" in payload and isinstance(payload["model"], Mapping):
            state = payload["model"]
        else:
            state = payload
        base_model = model.model if isinstance(model, ACTLossHead) else model
        base_model.load_state_dict(_normalize_ptrm_base_state_dict(state), strict=True)
        return payload

    def training_forward(
        self,
        model: nn.Module,
        state: object,
        batch: Mapping[str, Tensor],
    ) -> TrainingOutput:
        base_model = model.model if isinstance(model, ACTLossHead) else model
        if not isinstance(base_model, TinyRecursiveReasoningModel_ACTV1):
            raise TypeError("PTRMAdapter requires TinyRecursiveReasoningModel")
        new_carry, native = base_model(carry=state, batch=dict(batch))
        labels = new_carry.current_data["labels"]
        valid = labels != IGNORE_LABEL_ID
        counts = valid.sum(dim=-1).clamp_min(1)
        token_loss = stablemax_cross_entropy(
            native["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid
        )
        task_loss = aggregate_token_loss(
            token_loss,
            valid,
            mode=str(getattr(base_model, "_rrm_success_loss_aggregation", "mean")),
            smooth_delta=float(
                getattr(base_model, "_rrm_smooth_weakest_delta", math.log(2.0))
            ),
        )
        predictions = native["logits"].argmax(dim=-1)
        sequence_correct = ((predictions == labels) | ~valid).all(dim=-1)
        q_metrics: dict[str, Tensor]
        if base_model.config.cellwise_q_loss:
            q_terms = cellwise_q_loss_terms(
                prefix_logits=native["q_halt_logits"],
                cell_logits=native["q_cell_logits"],
                predictions=predictions,
                labels=labels,
                valid_mask=valid,
            )
            q_halt_loss = q_terms.halt_loss.sum()
            q_metrics = {
                "q_prefix_loss": q_terms.prefix_loss.sum().detach(),
                "q_cell_loss": q_terms.cell_loss.sum().detach(),
                "q_aggregate_loss": q_terms.aggregate_loss.sum().detach(),
            }
        else:
            q_halt_loss = F.binary_cross_entropy_with_logits(
                native["q_halt_logits"],
                sequence_correct.to(native["q_halt_logits"].dtype),
                reduction="sum",
            )
            q_metrics = {}
        q_continue_loss = native["q_halt_logits"].new_zeros(())
        if "target_q_continue" in native:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                native["q_continue_logits"],
                native["target_q_continue"],
                reduction="sum",
            )
        auxiliary_loss = float(getattr(base_model, "_rrm_q_loss_coeff", 0.5)) * (
            q_halt_loss + q_continue_loss
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
                **q_metrics,
            },
        )

    @torch.no_grad()
    def evaluation_candidates(
        self,
        model: nn.Module,
        batch: Mapping[str, Tensor],
        *,
        candidate_count: int | None = None,
        latent_noise_sigma: float | None = None,
        inference_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        base_model = model.model if isinstance(model, ACTLossHead) else model
        if not isinstance(base_model, TinyRecursiveReasoningModel_ACTV1):
            raise TypeError("PTRMAdapter requires TinyRecursiveReasoningModel")
        was_training = base_model.training
        base_model.eval()
        candidates = int(
            getattr(base_model, "_rrm_candidate_count", 1)
            if candidate_count is None
            else candidate_count
        )
        sigma = float(
            getattr(base_model, "_rrm_latent_noise_sigma", 0.0)
            if latent_noise_sigma is None
            else latent_noise_sigma
        )
        depth = int(
            getattr(base_model, "_rrm_inference_depth", base_model.config.halt_max_steps)
            if inference_depth is None
            else inference_depth
        )
        if candidates < 1 or depth < 1 or sigma < 0:
            raise ValueError("candidate count/depth must be positive and sigma non-negative")
        native_batch = {
            key: value.repeat_interleave(candidates, dim=0)
            for key, value in batch.items()
        }
        batch_size = int(batch["inputs"].shape[0])
        flat_size = batch_size * candidates
        device = batch["inputs"].device
        with torch.device(device):
            carry = base_model.inner.empty_carry(flat_size)
        carry = base_model.inner.reset_carry(
            torch.ones(flat_size, dtype=torch.bool, device=device), carry
        )
        logits: Tensor | None = None
        q_halt: Tensor | None = None
        for _ in range(depth):
            if sigma:
                carry = add_latent_noise(carry, sigma=sigma, generator=generator)
            carry, logits, q_outputs = base_model.inner(carry, native_batch)
            q_halt = q_outputs[0]
        if logits is None or q_halt is None:
            raise RuntimeError("PTRM recurrent rollout produced no output")
        candidate_predictions = logits.argmax(dim=-1).view(
            batch_size, candidates, -1
        )
        candidate_scores = q_halt.float().view(batch_size, candidates)
        if was_training:
            base_model.train()
        return candidate_predictions, candidate_scores

    @torch.no_grad()
    def evaluation_forward(
        self, model: nn.Module, batch: Mapping[str, Tensor]
    ) -> EvaluationOutput:
        base_model = model.model if isinstance(model, ACTLossHead) else model
        candidate_predictions, candidate_scores = self.evaluation_candidates(
            model, batch
        )
        batch_size = int(batch["inputs"].shape[0])
        depth = int(
            getattr(base_model, "_rrm_inference_depth", base_model.config.halt_max_steps)
        )
        device = batch["inputs"].device
        selected = select_first_max(candidate_scores)
        rows = torch.arange(batch_size, device=device)
        predictions = candidate_predictions[rows, selected]
        selection_score = candidate_scores[rows, selected]
        return EvaluationOutput(
            predictions=predictions,
            selection_score=selection_score,
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            steps=torch.full(
                (batch_size,), depth, dtype=torch.int32, device=device
            ),
        )

    def build_optimizers(
        self, model: nn.Module, preset: Mapping[str, Any]
    ) -> tuple[Optimizer, Optimizer | None]:
        base_model = model.model if isinstance(model, ACTLossHead) else model
        dense_parameters = [
            parameter
            for parameter in base_model.parameters()
            if parameter.requires_grad
        ]
        dense = torch.optim.AdamW(
            dense_parameters,
            lr=float(preset.get("learning_rate", preset.get("lr", 1e-4))),
            weight_decay=float(preset.get("weight_decay", 1e-4)),
            betas=(
                float(preset.get("beta1", 0.9)),
                float(preset.get("beta2", 0.999)),
            ),
        )
        sparse: Optimizer | None = None
        if getattr(base_model.config, "puzzle_emb_ndim", 0) > 0:
            sparse = CastedSparseEmbeddingSignSGD_Distributed(
                base_model.puzzle_emb.buffers(),
                world_size=int(preset.get("world_size", 1)),
                lr=float(preset.get("puzzle_embedding_learning_rate", 0.01)),
                weight_decay=float(
                    preset.get("puzzle_embedding_weight_decay", 1.0)
                ),
            )
        return dense, sparse


ADAPTER = PTRMAdapter()
