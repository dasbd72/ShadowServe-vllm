# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x–style decoder blocks (RMSNorm, GQA, SwiGLU) for shadow CPU execution.

Parallel structure to ``vllm.model_executor.models.llama`` (``LlamaMLP``,
``LlamaAttention``, ``LlamaDecoderLayer``, ``LlamaModel``, ``LlamaForCausalLM``)
using ``nn.Module``.
Linear projections use shadow ``Linear`` (HF-shaped weights; oneDNN plan built
inside the module).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from vllm.shadow.models import cpu_ops
from vllm.shadow.models.hf_config import ShadowHfConfig
from vllm.shadow.models.kv_state import ShadowPagedAttentionBatch
from vllm.shadow.models.linear import Linear
from vllm.shadow.models.rotary_embedding import build_shadow_cos_sin_cache

logger = logging.getLogger(__name__)


def _strip_model_prefix(name: str) -> str:
    return name[6:] if name.startswith("model.") else name


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    cpu_ops.rms_norm(out, x, weight, eps)
    return out


def _cpu_attn_sliding_window_args(hf: ShadowHfConfig) -> tuple[int, int, int]:
    """``(left, right, scheduler_window_size)`` for CPU attention.

    Matches ``LlamaModel``.
    """
    if hf.sliding_window is None:
        return -1, -1, -1
    w = int(hf.sliding_window)
    return w - 1, 0, w


@dataclass(slots=True)
class LlamaLayerWeight:
    """Per-layer tensors for a decoder block.

    Attention/MLP weights go to ``LlamaAttention`` / ``LlamaMLP``.
    """

    attn_norm: torch.Tensor
    qkv_proj: torch.Tensor
    o_proj: torch.Tensor
    ffn_norm: torch.Tensor
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    qkv_proj_bias: torch.Tensor | None = None
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None
    o_proj_bias: torch.Tensor | None = None
    gate_up_bias: torch.Tensor | None = None
    down_proj_bias: torch.Tensor | None = None


class LlamaMLP(nn.Module):
    """CPU SwiGLU FFN: merged gate+up projection, SiLU*gate, row down.

    Mirrors GPU ``LlamaMLP``.
    """

    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        *,
        gate_up_bias: torch.Tensor | None = None,
        down_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        gate_up_out, gate_up_in = gate_up_weight.shape
        down_out, down_in = down_weight.shape
        self.gate_up_proj = Linear(
            gate_up_in, gate_up_out, gate_up_weight, gate_up_bias
        )
        self.down_proj = Linear(down_in, down_out, down_weight, down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        inter = torch.empty(
            (x.size(0), self.down_proj.in_features),
            dtype=x.dtype,
            device=x.device,
        )
        cpu_ops.silu_and_mul(inter, gate_up)
        return self.down_proj(inter)


class LlamaAttention(nn.Module):
    """QKV projection, optional Q/K norm, RoPE, paged KV attention.

    Includes output projection.
    """

    def __init__(
        self,
        layer_idx: int,
        hf: ShadowHfConfig,
        cos_sin_cache: torch.Tensor,
        isa: str,
        qkv_weight: torch.Tensor,
        o_weight: torch.Tensor,
        *,
        qkv_bias: torch.Tensor | None = None,
        o_bias: torch.Tensor | None = None,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        sw_l, sw_r, _ = _cpu_attn_sliding_window_args(hf)
        self.sliding_window_left = sw_l
        self.sliding_window_right = sw_r

        self.rms_norm_eps = hf.rms_norm_eps
        self.num_attention_heads = hf.num_attention_heads
        self.num_key_value_heads = hf.num_key_value_heads
        self.head_dim = hf.head_dim
        self.q_size = hf.num_attention_heads * hf.head_dim
        self.kv_size = hf.num_key_value_heads * hf.head_dim
        self.scale = float(hf.head_dim**-0.5)

        self.isa = isa
        self.register_buffer("cos_sin_cache", cos_sin_cache)
        qkv_out, qkv_in = qkv_weight.shape
        self.qkv_proj = Linear(qkv_in, qkv_out, qkv_weight, qkv_bias)
        o_out, o_in = o_weight.shape
        self.o_proj = Linear(o_in, o_out, o_weight, o_bias)
        if q_norm is not None:
            self.register_buffer("q_norm", q_norm)
        else:
            self.q_norm = None
        if k_norm is not None:
            self.register_buffer("k_norm", k_norm)
        else:
            self.k_norm = None

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Qwen3 / Qwen2.5-style per-head RMSNorm on Q and K (before RoPE)."""
        eps = self.rms_norm_eps
        if self.q_norm is None or self.k_norm is None:
            return q, k
        b, nh, hd = q.shape
        q_flat = q.reshape(b * nh, hd)
        q_out = _rms_norm(q_flat, self.q_norm, eps)
        q = q_out.view(b, nh, hd)
        b2, nkv, hd2 = k.shape
        if hd2 != hd:
            raise ValueError("k head_dim mismatch vs q")
        k_flat = k.reshape(b2 * nkv, hd)
        k_out = _rms_norm(k_flat, self.k_norm, eps)
        k = k_out.view(b2, nkv, hd)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        batch: ShadowPagedAttentionBatch,
        scheduler_metadata: object,
    ) -> torch.Tensor:
        # Project hidden states to QKV
        out = self.qkv_proj(hidden_states)
        q = out[:, : self.q_size].view(-1, self.num_attention_heads, self.head_dim)
        k = out[:, self.q_size : self.q_size + self.kv_size].view(
            -1, self.num_key_value_heads, self.head_dim
        )
        v = out[:, self.q_size + self.kv_size :].view(
            -1, self.num_key_value_heads, self.head_dim
        )
        # Apply QK normalization if enabled (before RoPE)
        q, k = self._apply_qk_norm(q, k)
        q_rot = q.reshape(q.size(0), -1)
        k_rot = k.reshape(k.size(0), -1)
        # Apply RoPE
        cpu_ops.rotary_embedding(
            positions,
            q_rot,
            k_rot,
            self.head_dim,
            self.cos_sin_cache,
            True,
        )
        # Apply attention with KV cache
        k_cache, v_cache = batch.kv_caches[self.layer_idx]
        attn_out = torch.empty_like(q)
        cpu_ops.cpu_attn_reshape_and_cache(
            k,
            v,
            k_cache,
            v_cache,
            batch.slot_mapping,
            self.isa,
        )
        cpu_ops.cpu_attention_with_kv_cache(
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            output=attn_out,
            query_start_loc=batch.query_start_loc,
            seq_lens=batch.seq_lens,
            scale=self.scale,
            causal=True,
            alibi_slopes=None,
            sliding_window_left=self.sliding_window_left,
            sliding_window_right=self.sliding_window_right,
            block_table=batch.block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )
        # Project attention output to output
        attn_flat = attn_out.reshape(-1, self.num_attention_heads * self.head_dim)
        return self.o_proj(attn_flat)


class LlamaDecoderLayer(nn.Module):
    """One transformer block: pre-norm attention, post-norm MLP (matches GPU order)."""

    def __init__(
        self,
        layer_idx: int,
        hf: ShadowHfConfig,
        cos_sin_cache: torch.Tensor,
        isa: str,
        weight: LlamaLayerWeight,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.rms_norm_eps = hf.rms_norm_eps

        self.register_buffer("attn_norm", weight.attn_norm)
        self.register_buffer("ffn_norm", weight.ffn_norm)

        self.self_attn = LlamaAttention(
            layer_idx,
            hf,
            cos_sin_cache,
            isa,
            weight.qkv_proj,
            weight.o_proj,
            qkv_bias=weight.qkv_proj_bias,
            o_bias=weight.o_proj_bias,
            q_norm=weight.q_norm,
            k_norm=weight.k_norm,
        )
        self.mlp = LlamaMLP(
            weight.gate_up_proj,
            weight.down_proj,
            gate_up_bias=weight.gate_up_bias,
            down_bias=weight.down_proj_bias,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        batch: ShadowPagedAttentionBatch,
        scheduler_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.rms_norm_eps
        if residual is None:
            residual = hidden_states
            hidden_states = _rms_norm(hidden_states, self.attn_norm, eps)
        else:
            cpu_ops.fused_add_rms_norm(hidden_states, residual, self.attn_norm, eps)
        hidden_states = self.self_attn(
            positions, hidden_states, batch, scheduler_metadata
        )
        cpu_ops.fused_add_rms_norm(hidden_states, residual, self.ffn_norm, eps)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    """Transformer stack: ``embed_tokens`` + ``layers`` + final norm (no LM head)."""

    def __init__(
        self,
        hf: ShadowHfConfig,
        dtype: torch.dtype,
        embed_tokens: torch.Tensor,
        norm: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_specs: list[LlamaLayerWeight],
        isa: str,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("embed_tokens", embed_tokens)
        self.register_buffer("norm", norm)
        self.register_buffer("cos_sin_cache", cos_sin_cache)
        self.isa = isa
        self.num_attention_heads = hf.num_attention_heads
        self.num_key_value_heads = hf.num_key_value_heads
        self.rms_norm_eps = hf.rms_norm_eps
        self.head_dim = hf.head_dim
        self.scale = float(self.head_dim**-0.5)
        sw_l, sw_r, sched_w = _cpu_attn_sliding_window_args(hf)
        self.sliding_window_left = sw_l
        self.sliding_window_right = sw_r
        self.scheduler_window_arg = sched_w

        self.layers = nn.ModuleList(
            LlamaDecoderLayer(i, hf, self.cos_sin_cache, self.isa, weight)
            for i, weight in enumerate(layer_specs)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.embed_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch: ShadowPagedAttentionBatch,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder layers + final RMSNorm; returns hidden states before LM head."""
        dtype = self.dtype
        num_reqs = int(batch.block_table.size(0))
        scheduler_metadata = cpu_ops.cpu_attn_get_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            seq_lens=batch.seq_lens,
            dtype=dtype,
            query_start_loc=batch.query_start_loc,
            causal=True,
            sliding_window_size=self.scheduler_window_arg,
            isa=self.isa,
            enable_kv_split=True,
        )
        eps = self.rms_norm_eps
        h = hidden_states
        residual: torch.Tensor | None = None
        if not self.layers:
            return _rms_norm(h, self.norm, eps)
        for layer in self.layers:
            h, residual = layer(positions, h, residual, batch, scheduler_metadata)
        cpu_ops.fused_add_rms_norm(h, residual, self.norm, eps)
        return h


class LlamaForCausalLM(nn.Module):
    """Llama-style CPU decoder: inner ``LlamaModel`` + LM head."""

    def __init__(
        self,
        hf: ShadowHfConfig,
        dtype: torch.dtype,
        embed_tokens: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_specs: list[LlamaLayerWeight],
        eos_token_id: int | None,
        isa: str,
    ) -> None:
        super().__init__()
        self.hf = hf
        self.eos_token_id = eos_token_id
        self.model = LlamaModel(
            hf,
            dtype,
            embed_tokens,
            norm,
            cos_sin_cache,
            layer_specs,
            isa,
        )
        lm_head_out, lm_head_in = lm_head_weight.shape
        self.lm_head = Linear(lm_head_in, lm_head_out, lm_head_weight, None)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(self, batch: ShadowPagedAttentionBatch) -> torch.Tensor:
        """Last-position logits per request.

        Shape ``[len(request_indices), vocab]``.
        """
        if not batch.query_token_ids:
            raise ValueError("empty batch")

        device = self.model.embed_tokens.device
        query_lens = [len(ch) for ch in batch.query_token_ids]
        if any(L <= 0 for L in query_lens):
            raise ValueError("each query chunk must be non-empty")

        flat_ids: list[int] = []
        for chunk in batch.query_token_ids:
            flat_ids.extend(chunk)
        ids = torch.tensor(flat_ids, dtype=torch.long, device=device)
        h = self.model.embed_input_ids(ids)

        positions = batch.positions.to(device=h.device, dtype=torch.long)
        h = self.model(h, batch, positions)
        logits_indices = batch.query_start_loc[1:] - 1
        sample_hidden_states = h[logits_indices]
        return self.compute_logits(sample_hidden_states)


def _pick(
    weights: dict[str, torch.Tensor],
    *candidates: str,
) -> torch.Tensor:
    for c in candidates:
        if c in weights:
            return weights.pop(c)
    raise KeyError(f"missing weight, tried: {candidates}")


def _load_qkv_weights(w: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    q = _pick(w, f"layers.{layer_idx}.self_attn.q_proj.weight")
    k = _pick(w, f"layers.{layer_idx}.self_attn.k_proj.weight")
    v = _pick(w, f"layers.{layer_idx}.self_attn.v_proj.weight")
    return torch.cat([q, k, v], dim=0)


def _load_qkv_bias(w: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor | None:
    """Fused QKV bias (Qwen2-style: all three present or all omitted)."""
    qb, kb, vb = (
        f"layers.{layer_idx}.self_attn.q_proj.bias",
        f"layers.{layer_idx}.self_attn.k_proj.bias",
        f"layers.{layer_idx}.self_attn.v_proj.bias",
    )
    present = sum(1 for k in (qb, kb, vb) if k in w)
    if present == 0:
        return None
    if present != 3:
        raise KeyError(
            f"layer {layer_idx}: expected all or none of {qb}, {kb}, {vb} "
            f"(found {present}/3)"
        )
    q = _pick(w, qb)
    k = _pick(w, kb)
    v = _pick(w, vb)
    return torch.cat([q, k, v], dim=0)


def _load_gate_up_weights(w: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    gate = _pick(w, f"layers.{layer_idx}.mlp.gate_proj.weight")
    up = _pick(w, f"layers.{layer_idx}.mlp.up_proj.weight")
    return torch.cat([gate, up], dim=0)


def _load_gate_up_bias(
    w: dict[str, torch.Tensor], layer_idx: int
) -> torch.Tensor | None:
    """Merged gate+up bias (``MergedColumnParallelLinear`` / HF split tensors)."""
    gb = f"layers.{layer_idx}.mlp.gate_proj.bias"
    ub = f"layers.{layer_idx}.mlp.up_proj.bias"
    present = sum(1 for k in (gb, ub) if k in w)
    if present == 0:
        return None
    if present != 2:
        raise KeyError(
            f"layer {layer_idx}: expected both or neither of {gb} and {ub} "
            f"(found {present}/2)"
        )
    g = _pick(w, gb)
    u = _pick(w, ub)
    return torch.cat([g, u], dim=0)


def _load_qk_norm_weights(
    w: dict[str, torch.Tensor], layer_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-head Q/K RMSNorm weights (Qwen3 / optional Qwen2 ``qk_norm``)."""
    qn_path = f"layers.{layer_idx}.self_attn.q_norm.weight"
    kn_path = f"layers.{layer_idx}.self_attn.k_norm.weight"
    present = sum(1 for k in (qn_path, kn_path) if k in w)
    if present == 0:
        return None, None
    if present != 2:
        raise KeyError(
            f"layer {layer_idx}: expected both {qn_path} and {kn_path} for QK norm "
            f"(found {present}/2)"
        )
    q_norm = _pick(w, qn_path)
    k_norm = _pick(w, kn_path)
    return q_norm, k_norm


def build_llama_for_causal_lm(
    hf: ShadowHfConfig,
    weights: dict[str, torch.Tensor],
    dtype: torch.dtype,
    *,
    block_size: int,
) -> LlamaForCausalLM:
    """Build from HF weight iterator."""
    w: dict[str, torch.Tensor] = {}
    device = torch.device("cpu")
    for name in list(weights.keys()):
        n = _strip_model_prefix(name)
        w[n] = torch.empty_like(weights[name], device=device, dtype=dtype)
        w[n].copy_(weights[name])
        weights.pop(name)

    if not cpu_ops.supports_onednn():
        raise RuntimeError("shadow LlamaForCausalLM requires oneDNN CPU matmul.")

    embed = _pick(w, "embed_tokens.weight")
    lm_head = w.get("lm_head.weight", embed)

    norm = _pick(w, "norm.weight")

    cos_sin = build_shadow_cos_sin_cache(hf, dtype=dtype, device=device)

    layer_specs: list[LlamaLayerWeight] = []
    L = hf.num_hidden_layers
    for i in range(L):
        qkv = _load_qkv_weights(w, i)
        o = _pick(w, f"layers.{i}.self_attn.o_proj.weight")
        o_bias = w.get(f"layers.{i}.self_attn.o_proj.bias", None)
        qkv_b = _load_qkv_bias(w, i)
        gate_up = _load_gate_up_weights(w, i)
        gate_up_b = _load_gate_up_bias(w, i)
        down = _pick(w, f"layers.{i}.mlp.down_proj.weight")
        down_b = w.get(f"layers.{i}.mlp.down_proj.bias", None)
        attn_norm = _pick(w, f"layers.{i}.input_layernorm.weight")
        ffn_norm = _pick(w, f"layers.{i}.post_attention_layernorm.weight")
        q_norm, k_norm = _load_qk_norm_weights(w, i)
        layer_specs.append(
            LlamaLayerWeight(
                attn_norm=attn_norm,
                qkv_proj=qkv,
                o_proj=o,
                ffn_norm=ffn_norm,
                gate_up_proj=gate_up,
                down_proj=down,
                qkv_proj_bias=qkv_b,
                q_norm=q_norm,
                k_norm=k_norm,
                o_proj_bias=o_bias,
                gate_up_bias=gate_up_b,
                down_proj_bias=down_b,
            )
        )

    w.clear()

    isa = cpu_ops.shadow_cpu_attn_isa(dtype, int(block_size), hf.head_dim)
    return LlamaForCausalLM(
        hf,
        dtype,
        embed,
        lm_head,
        norm,
        cos_sin,
        layer_specs,
        hf.eos_token_id,
        isa,
    )


__all__ = (
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaForCausalLM",
    "LlamaLayerWeight",
    "LlamaMLP",
    "LlamaModel",
    "build_llama_for_causal_lm",
)
