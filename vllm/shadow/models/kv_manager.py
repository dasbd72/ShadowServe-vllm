# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged KV state for the shadow CPU executor (no engine/scheduler imports)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch

from vllm.shadow.transfer.kvhtc_memfd import MemfdTensor
from vllm.shadow.transfer.kvhtc_protocol import (
    MigrationHandoff,
    torch_dtype_from_str,
)

logger = logging.getLogger("vllm.shadow.executor.shadow_kv_manager")

__all__ = ("ShadowKvManager",)


def _required_blocks(num_computed_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_computed_tokens <= 0:
        return 0
    return (num_computed_tokens + block_size - 1) // block_size


@dataclass(slots=True)
class ShadowPagedAttentionBatch:
    """Batch metadata for CPU paged attention (unified decode + prefill).

    ``query_token_ids`` has one list per scheduled request; each list length is
    the query length for that row (see :meth:`ShadowKvManager.build_attention_batch`).

    ``kv_caches`` is the per-layer K/V tensor list.

    ``slot_mapping`` has length ``sum(len(ids) for ids in query_token_ids)`` in
    request order (same token order as the flattened hidden states / QKV tensors).
    ``seq_lens`` has length ``len(request_indices)`` — the CPU kernel indexes KV
    length and block tables **per request**, not per token.

    ``positions`` has the same flattened length; for each
    request row, token positions are ``num_computed .. num_computed + query_len - 1``
    (from :attr:`ShadowKvManager` state at batch build time).
    """

    query_token_ids: list[list[int]]
    kv_caches: list[torch.Tensor]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    positions: torch.Tensor
    max_seq_len: int


class ShadowKvManager:
    """Validates migration handoff, holds per-layer memfd KV tensors, tracks
    per-request block tables for incremental decode, and per-request prompt vs
    output token ids (shadow decode appends to the output lists)."""

    __slots__ = (
        "_handoff",
        "_layers",
        "_block_tables",
        "_num_computed",
        "_prompt_token_ids",
        "_output_token_ids",
        "_free_blocks",
        "_expected_dtype",
        "_expected_shape",
    )

    def __init__(self, handoff: MigrationHandoff) -> None:
        self._handoff: Final = handoff

        assigned: set[int] = set()
        for ri, req in enumerate(handoff.requests):
            for bid in req.dst_block_table:
                if bid < 0 or bid >= handoff.shadow_num_blocks:
                    raise ValueError(
                        f"request {ri}: dst_block_table entry {bid} out of range "
                        f"[0, {handoff.shadow_num_blocks})"
                    )
                if bid in assigned:
                    raise ValueError(
                        f"request {ri}: duplicate physical block {bid} across handoff"
                    )
                assigned.add(bid)
            need = _required_blocks(req.num_computed_tokens, handoff.block_size)
            if len(req.dst_block_table) < need:
                raise ValueError(
                    f"request {ri}: len(dst_block_table)={len(req.dst_block_table)} "
                    f"< required {need} for num_computed_tokens="
                    f"{req.num_computed_tokens} block_size={handoff.block_size}"
                )

        self._block_tables = [list(r.dst_block_table) for r in handoff.requests]
        self._num_computed = [int(r.num_computed_tokens) for r in handoff.requests]
        self._prompt_token_ids: list[list[int]] = []
        self._output_token_ids: list[list[int]] = []
        for r in handoff.requests:
            prompt = list(r.prompt_token_ids) if r.prompt_token_ids is not None else []
            output = list(r.output_token_ids)
            self._prompt_token_ids.append(prompt)
            self._output_token_ids.append(output)

        self._free_blocks: list[int] = [
            i for i in range(handoff.shadow_num_blocks) if i not in assigned
        ]

        self._expected_dtype = torch_dtype_from_str(handoff.dtype)
        self._expected_shape = (
            2,
            handoff.shadow_num_blocks,
            handoff.num_kv_heads,
            handoff.block_size,
            handoff.head_dim,
        )

        self._layers: list[MemfdTensor | None] = [None] * handoff.num_layers

        logger.debug(
            "ShadowKvManager migration_id=%s shadow_num_blocks=%s free_blocks=%d",
            handoff.migration_id,
            handoff.shadow_num_blocks,
            len(self._free_blocks),
        )

    def register_layer(
        self,
        layer_idx: int,
        tensor: MemfdTensor,
    ) -> None:
        if layer_idx < 0 or layer_idx >= self._handoff.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        if self._layers[layer_idx] is not None:
            raise ValueError(f"KV already registered for layer {layer_idx}")
        t = tensor.tensor.detach()
        if t.device.type != "cpu":
            raise ValueError("layer KV tensor must be on CPU")
        if not t.is_contiguous():
            raise ValueError("layer KV tensor must be contiguous")
        if tuple(t.shape) != self._expected_shape:
            raise ValueError(
                f"layer KV shape {tuple(t.shape)} != expected {self._expected_shape}"
            )
        if t.dtype != self._expected_dtype:
            raise ValueError(
                f"layer KV dtype {t.dtype} != expected {self._expected_dtype}"
            )
        self._layers[layer_idx] = tensor

    def num_computed_tokens(self, req_index: int) -> int:
        if req_index < 0 or req_index >= len(self._num_computed):
            raise ValueError(f"req_index out of range: {req_index}")
        return int(self._num_computed[req_index])

    def prompt_len(self, req_index: int) -> int:
        if req_index < 0 or req_index >= len(self._prompt_token_ids):
            raise ValueError(f"req_index out of range: {req_index}")
        return len(self._prompt_token_ids[req_index])

    def num_output_tokens(self, req_index: int) -> int:
        if req_index < 0 or req_index >= len(self._output_token_ids):
            raise ValueError(f"req_index out of range: {req_index}")
        return len(self._output_token_ids[req_index])

    def uncomputed_token_ids(self, req_index: int) -> list[int]:
        """Token ids not yet written to KV (suffix of prompt+output in order)."""
        if req_index < 0 or req_index >= len(self._prompt_token_ids):
            raise ValueError(f"req_index out of range: {req_index}")
        prompt = self._prompt_token_ids[req_index]
        output = self._output_token_ids[req_index]
        nc = self._num_computed[req_index]
        p_len = len(prompt)
        if nc < p_len:
            return prompt[nc:] + output
        return output[nc - p_len :]

    def append_decoded_token(self, req_index: int, token_id: int) -> None:
        """Append one shadow-decoded token to the request output sequence."""
        if req_index < 0 or req_index >= len(self._output_token_ids):
            raise ValueError(f"req_index out of range: {req_index}")
        self._output_token_ids[req_index].append(token_id)

    def _ensure_block_for_logical(self, req_index: int, logical_block_idx: int) -> None:
        """Ensure ``_block_tables[req_index]`` covers ``logical_block_idx``."""
        if req_index < 0 or req_index >= len(self._block_tables):
            raise ValueError(f"req_index out of range: {req_index}")
        if logical_block_idx < 0:
            raise ValueError("logical_block_idx must be non-negative")
        table = self._block_tables[req_index]
        while logical_block_idx >= len(table):
            if not self._free_blocks:
                raise RuntimeError(
                    "no free shadow KV slots left for block table extension "
                    f"(req_index={req_index} logical_block_idx={logical_block_idx})"
                )
            table.append(self._free_blocks.pop())

    def prepare_blocks_for_query(self, req_index: int, num_query_tokens: int) -> None:
        """Allocate physical blocks for the next KV write range for this request."""
        if num_query_tokens <= 0:
            raise ValueError("num_query_tokens must be positive")
        if req_index < 0 or req_index >= len(self._num_computed):
            raise ValueError(f"req_index out of range: {req_index}")
        start = self._num_computed[req_index]
        for pos in range(start, start + num_query_tokens):
            self._ensure_block_for_logical(req_index, pos // self._handoff.block_size)

    def advance_num_computed(self, req_index: int, delta: int = 1) -> None:
        """Increment cached sequence length after KV is written."""
        if delta <= 0:
            raise ValueError("delta must be positive")
        if req_index < 0 or req_index >= len(self._num_computed):
            raise ValueError(f"req_index out of range: {req_index}")
        self._num_computed[req_index] += delta

    def build_attention_batch(
        self,
        request_indices: list[int],
        query_token_ids: list[list[int]],
    ) -> ShadowPagedAttentionBatch:
        """Build unified batch metadata from per-request query token id lists.

        Decode is the special case where every inner list has length 1.
        """
        if not request_indices:
            raise ValueError("empty batch")
        if len(query_token_ids) != len(request_indices):
            raise ValueError("query_token_ids and request_indices mismatch")

        query_lens = [len(ch) for ch in query_token_ids]
        if any(L <= 0 for L in query_lens):
            raise ValueError("each query_token_ids chunk must be non-empty")

        handoff = self._handoff
        block_size = int(handoff.block_size)
        tables = self._block_tables
        num_computed = self._num_computed

        n = len(request_indices)
        max_blocks = max(len(tables[ri]) for ri in request_indices)
        block_table = torch.zeros((n, max_blocks), dtype=torch.int32)

        total_tokens = sum(query_lens)
        slot_mapping = torch.empty((total_tokens,), dtype=torch.int64)
        pos_parts: list[torch.Tensor] = []
        seq_after: list[int] = []
        sm = 0
        for row, (ri, L) in enumerate(zip(request_indices, query_lens, strict=True)):
            blks = tables[ri]
            nb = len(blks)
            if nb > max_blocks:
                raise RuntimeError("internal: max_blocks too small")
            block_table[row, :nb] = torch.tensor(blks, dtype=torch.int32)
            start = int(num_computed[ri])
            seq_after.append(start + L)
            pos_parts.append(
                torch.arange(
                    start, start + L, dtype=torch.int64, device=block_table.device
                )
            )
            for j in range(L):
                pos = start + j
                logical_block = pos // block_size
                if logical_block >= len(blks):
                    raise RuntimeError(
                        "block table too short for position "
                        f"(req={ri} pos={pos} logical_block={logical_block})"
                    )
                phys = int(blks[logical_block])
                slot_mapping[sm + j] = phys * block_size + (pos % block_size)
            sm += L

        seq_lens = torch.tensor(seq_after, dtype=torch.int32)
        lens_t = torch.tensor(query_lens, dtype=torch.int32)
        qsl = torch.zeros((n + 1,), dtype=torch.int32)
        qsl[1:] = torch.cumsum(lens_t, dim=0)
        positions = torch.cat(pos_parts, dim=0)
        max_seq_len = int(seq_lens.max().item())
        return ShadowPagedAttentionBatch(
            query_token_ids=query_token_ids,
            kv_caches=self._layer_kv_caches(),
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            positions=positions,
            max_seq_len=max_seq_len,
        )

    def _layer_kv_caches(self) -> list[torch.Tensor]:
        """K/V per layer, shape ``[num_blocks, num_kv_heads, block_size, head_dim]``.

        **K** uses the packed tile layout from ``gather_kv_blocks`` /
        ``cpu_attn_reshape_and_cache``; **V** is row-major ``(slot, dim)`` in the
        last two axes (HND / flash-style).
        """
        kv_caches: list[torch.Tensor] = []
        for i, tensor in enumerate(self._layers):
            if tensor is None:
                raise ValueError(f"layer {i} is not registered yet")
            kv_caches.append(tensor.tensor)
        return kv_caches
