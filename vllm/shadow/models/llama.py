# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x–style decoder blocks (RMSNorm, GQA, SwiGLU) for shadow CPU execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.shadow.models import cpu_ops
from vllm.shadow.models.hf_config import ShadowHfConfig
from vllm.shadow.models.kv_state import ShadowPagedAttentionBatch
from vllm.shadow.models.rotary_embedding import build_shadow_cos_sin_cache


def _strip_model_prefix(name: str) -> str:
    return name[6:] if name.startswith("model.") else name


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    cpu_ops.rms_norm(out, x, weight, eps)
    return out


def _linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


@dataclass(slots=True)
class LlamaLayerWeights:
    attn_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    ffn_norm: torch.Tensor
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    q_proj_bias: torch.Tensor | None = None
    k_proj_bias: torch.Tensor | None = None
    v_proj_bias: torch.Tensor | None = None
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None
    o_proj_bias: torch.Tensor | None = None
    gate_up_bias: torch.Tensor | None = None
    down_proj_bias: torch.Tensor | None = None


class LlamaLikeShadowModel:
    """Llama-style decoder on CPU with unified forward over paged KV."""

    __slots__ = (
        "hf",
        "dtype",
        "embed",
        "lm_head",
        "norm",
        "cos_sin_cache",
        "layers",
        "eos_token_id",
        "head_dim",
        "scale",
        "isa",
        "sliding_window_left",
        "sliding_window_right",
        "scheduler_window_arg",
    )

    def __init__(
        self,
        hf: ShadowHfConfig,
        dtype: torch.dtype,
        embed: torch.Tensor,
        lm_head: torch.Tensor,
        norm: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layers: list[LlamaLayerWeights],
        eos_token_id: int | None,
        isa: str,
    ) -> None:
        self.hf = hf
        self.dtype = dtype
        self.embed = embed
        self.lm_head = lm_head
        self.norm = norm
        self.cos_sin_cache = cos_sin_cache
        self.layers = layers
        self.eos_token_id = eos_token_id
        self.head_dim = hf.head_dim
        self.scale = float(self.head_dim**-0.5)
        self.isa = isa
        if hf.sliding_window is None:
            self.sliding_window_left = -1
            self.sliding_window_right = -1
            self.scheduler_window_arg = -1
        else:
            w = int(hf.sliding_window)
            self.sliding_window_left = w - 1
            self.sliding_window_right = 0
            self.scheduler_window_arg = w

    def _reshape_qkv(
        self,
        x: torch.Tensor,
        layer: LlamaLayerWeights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nh = self.hf.num_attention_heads
        nkv = self.hf.num_key_value_heads
        hd = self.head_dim
        q = _linear(x, layer.q_proj, layer.q_proj_bias).view(-1, nh, hd)
        k = _linear(x, layer.k_proj, layer.k_proj_bias).view(-1, nkv, hd)
        v = _linear(x, layer.v_proj, layer.v_proj_bias).view(-1, nkv, hd)
        return q, k, v

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: LlamaLayerWeights,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Qwen3 / Qwen2.5-style per-head RMSNorm on Q and K

        (matches vLLM order: before RoPE).
        """
        if layer.q_norm is None or layer.k_norm is None:
            return q, k
        b, nh, hd = q.shape
        q_flat = q.reshape(b * nh, hd)
        q_out = _rms_norm(q_flat, layer.q_norm, self.hf.rms_norm_eps)
        q = q_out.view(b, nh, hd)
        b2, nkv, hd2 = k.shape
        if hd2 != hd:
            raise ValueError("k head_dim mismatch vs q")
        k_flat = k.reshape(b2 * nkv, hd)
        k_out = _rms_norm(k_flat, layer.k_norm, self.hf.rms_norm_eps)
        k = k_out.view(b2, nkv, hd)
        return q, k

    def _compute_logits_from_embeddings(
        self,
        h: torch.Tensor,
        batch: ShadowPagedAttentionBatch,
    ) -> torch.Tensor:
        """Run transformer layers + final norm + LM head.

        logits shape ``[num_tokens, vocab]``.
        """
        positions = batch.positions.to(device=h.device, dtype=torch.long)
        dtype = self.dtype
        num_reqs = int(batch.block_table.size(0))
        scheduler_metadata = cpu_ops.cpu_attn_get_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=self.hf.num_attention_heads,
            num_kv_heads=self.hf.num_key_value_heads,
            head_dim=self.head_dim,
            seq_lens=batch.seq_lens,
            dtype=dtype,
            query_start_loc=batch.query_start_loc,
            causal=True,
            sliding_window_size=self.scheduler_window_arg,
            isa=self.isa,
            enable_kv_split=True,
        )
        nh = self.hf.num_attention_heads
        hd = self.head_dim
        eps = self.hf.rms_norm_eps
        residual: torch.Tensor | None = None
        if not self.layers:
            x = _rms_norm(h, self.norm, eps)
            return _linear(x, self.lm_head, None).float()
        for li, layer in enumerate(self.layers):
            # Match vLLM ``LlamaDecoderLayer`` / ``Qwen2DecoderLayer``
            # (RMSNorm + fused residual stream).
            if residual is None:
                residual = h
                x = _rms_norm(h, layer.attn_norm, eps)
            else:
                cpu_ops.fused_add_rms_norm(h, residual, layer.attn_norm, eps)
                x = h
            q, k, v = self._reshape_qkv(x, layer)
            q, k = self._apply_qk_norm(q, k, layer)
            # CPU rotary expects last dim = num_heads * head_size
            # (see csrc/cpu/pos_encoding.cpp).
            q_rot = q.reshape(q.size(0), -1)
            k_rot = k.reshape(k.size(0), -1)
            cpu_ops.rotary_embedding(
                positions,
                q_rot,
                k_rot,
                self.head_dim,
                self.cos_sin_cache,
                True,
            )
            k_cache, v_cache = batch.kv_caches[li]
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
            attn_flat = attn_out.reshape(-1, nh * hd)
            attn_hidden = _linear(attn_flat, layer.o_proj, layer.o_proj_bias)
            cpu_ops.fused_add_rms_norm(attn_hidden, residual, layer.ffn_norm, eps)
            h = self._mlp(layer, attn_hidden)
        cpu_ops.fused_add_rms_norm(h, residual, self.norm, eps)
        # Float32 logits: bf16/half argmax is unstable vs GPU and hurts greedy decode.
        return _linear(h, self.lm_head, None).float()

    def forward_logits(self, batch: ShadowPagedAttentionBatch) -> torch.Tensor:
        """Run a forward pass over scheduled query tokens, return last-position logits.

        Returns float32 logits ``[len(request_indices), vocab]``. Each request
        processes ``len(batch.query_token_ids[i])`` new tokens; Decode is the
        special case where every chunk has length 1.
        """
        if not batch.query_token_ids:
            raise ValueError("empty batch")

        device = self.embed.device
        query_lens = [len(ch) for ch in batch.query_token_ids]
        if any(L <= 0 for L in query_lens):
            raise ValueError("each query chunk must be non-empty")

        emb_parts: list[torch.Tensor] = []
        for chunk in batch.query_token_ids:
            ids = torch.tensor(chunk, dtype=torch.long, device=device)
            emb_parts.append(torch.nn.functional.embedding(ids, self.embed))
        h = torch.cat(emb_parts, dim=0)

        logits_flat = self._compute_logits_from_embeddings(h, batch)
        qsl = batch.query_start_loc
        last_idx = (qsl[1:] - 1).to(device=device, dtype=torch.int64)
        return logits_flat[last_idx]

    def _mlp(self, layer: LlamaLayerWeights, x: torch.Tensor) -> torch.Tensor:
        gate_up = _linear(x, layer.gate_up_proj, layer.gate_up_bias)
        inter = torch.empty(
            (x.size(0), layer.down_proj.size(1)),
            dtype=x.dtype,
            device=x.device,
        )
        cpu_ops.silu_and_mul(inter, gate_up)
        return _linear(inter, layer.down_proj, layer.down_proj_bias)


def _pick(
    weights: dict[str, torch.Tensor],
    *candidates: str,
) -> torch.Tensor:
    for c in candidates:
        if c in weights:
            return weights[c]
    raise KeyError(f"missing weight, tried: {candidates}")


def _load_qkv_biases(
    w: dict[str, torch.Tensor], p: str, layer_idx: int
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Load Q/K/V projection biases (Qwen2-style: all present or all omitted)."""
    qb, kb, vb = f"{p}q_proj.bias", f"{p}k_proj.bias", f"{p}v_proj.bias"
    present = sum(1 for k in (qb, kb, vb) if k in w)
    if present == 0:
        return None, None, None
    if present != 3:
        raise KeyError(
            f"layer {layer_idx}: expected all or none of {qb}, {kb}, {vb} "
            f"(found {present}/3)"
        )
    return w[qb], w[kb], w[vb]


def _load_gate_up_bias(
    w: dict[str, torch.Tensor], layer_idx: int
) -> torch.Tensor | None:
    """Merged gate+up bias (``MergedColumnParallelLinear`` / HF split tensors).

    Both or neither biases must be present.
    """
    gb = f"layers.{layer_idx}.mlp.gate_proj.bias"
    ub = f"layers.{layer_idx}.mlp.up_proj.bias"
    ng, nu = gb in w, ub in w
    if ng and nu:
        return torch.cat([w[gb], w[ub]], dim=0)
    if not ng and not nu:
        return None
    raise KeyError(
        f"layer {layer_idx}: expected both or neither of {gb} and {ub} "
        f"(gate={ng}, up={nu})"
    )


def _load_qk_norm_weights(
    w: dict[str, torch.Tensor], p: str, layer_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-head Q/K RMSNorm weights (Qwen3 / optional Qwen2 ``qk_norm``).

    Both or neither weights must be present.
    """
    qn_path = f"{p}q_norm.weight"
    kn_path = f"{p}k_norm.weight"
    nq, nk = qn_path in w, kn_path in w
    if nq and nk:
        return w[qn_path], w[kn_path]
    if not nq and not nk:
        return None, None
    raise KeyError(
        f"layer {layer_idx}: expected both {qn_path} and {kn_path} for QK norm "
        f"(q_norm={nq}, k_norm={nk})"
    )


def build_llama_like_model(
    hf: ShadowHfConfig,
    weights: dict[str, torch.Tensor],
    dtype: torch.dtype,
    *,
    block_size: int,
) -> LlamaLikeShadowModel:
    """Build from HF-style ``state_dict`` keys (``model.`` prefix optional)."""
    w: dict[str, torch.Tensor] = {_strip_model_prefix(k): v for k, v in weights.items()}
    device = torch.device("cpu")

    embed = _pick(w, "embed_tokens.weight")
    lm_head = w.get("lm_head.weight", embed)

    norm = _pick(w, "norm.weight")

    cos_sin = build_shadow_cos_sin_cache(hf, dtype=dtype, device=device)

    layers: list[LlamaLayerWeights] = []
    L = hf.num_hidden_layers
    for i in range(L):
        p = f"layers.{i}.self_attn."
        q = _pick(w, f"{p}q_proj.weight")
        k = _pick(w, f"{p}k_proj.weight")
        v = _pick(w, f"{p}v_proj.weight")
        o = _pick(w, f"{p}o_proj.weight")
        o_bias = w.get(f"{p}o_proj.bias", None)
        q_b, k_b, v_b = _load_qkv_biases(w, p, i)
        gate = _pick(w, f"layers.{i}.mlp.gate_proj.weight")
        up = _pick(w, f"layers.{i}.mlp.up_proj.weight")
        gate_up = torch.cat([gate, up], dim=0)
        gate_up_b = _load_gate_up_bias(w, i)
        down = _pick(w, f"layers.{i}.mlp.down_proj.weight")
        down_b = w.get(f"layers.{i}.mlp.down_proj.bias", None)
        attn_norm = _pick(w, f"layers.{i}.input_layernorm.weight")
        ffn_norm = _pick(w, f"layers.{i}.post_attention_layernorm.weight")
        q_norm, k_norm = _load_qk_norm_weights(w, p, i)
        layers.append(
            LlamaLayerWeights(
                attn_norm=attn_norm,
                q_proj=q,
                k_proj=k,
                v_proj=v,
                o_proj=o,
                ffn_norm=ffn_norm,
                gate_up_proj=gate_up,
                down_proj=down,
                q_proj_bias=q_b,
                k_proj_bias=k_b,
                v_proj_bias=v_b,
                q_norm=q_norm,
                k_norm=k_norm,
                o_proj_bias=o_bias,
                gate_up_bias=gate_up_b,
                down_proj_bias=down_b,
            )
        )

    isa = cpu_ops.shadow_cpu_attn_isa(dtype, int(block_size), hf.head_dim)
    return LlamaLikeShadowModel(
        hf,
        dtype,
        embed,
        lm_head,
        norm,
        cos_sin,
        layers,
        hf.eos_token_id,
        isa,
    )
