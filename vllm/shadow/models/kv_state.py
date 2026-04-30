# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged KV state for the shadow CPU executor.

This module is intentionally **shadow-local**: it provides the minimal data
structures needed by the shadow CPU path to ingest migrated KV blocks and
continue incremental decoding without importing the vLLM engine/scheduler stack.

Concepts
--------
- **Block table**: per-request mapping from *logical* KV blocks
  (``pos // block_size``) to *physical* cache blocks (an integer slot id).
- **num_computed**: per-request count of tokens whose KV has been written into
  the paged cache. It is used to derive the next query token positions for
  decode/prefill and to validate handoff consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vllm.shadow.models.sampling import ShadowSamplingParams
from vllm.shadow.transfer.kv_transport_common import MemfdTensor, torch_dtype_from_str

logger = logging.getLogger("vllm.shadow.models.kv_state")

__all__ = ("ShadowKvState", "Request", "ShadowPagedAttentionBatch")


@dataclass(slots=True)
class Request:
    """Mutable per-request state (internal to :class:`ShadowKvState`).

    Notes
    -----
    - ``prompt_token_ids`` + ``output_token_ids`` represents the logical token
      sequence in order.
    - ``num_computed`` counts how many tokens (prefix of that logical sequence)
      already have KV written in the paged cache.
    - ``block_table`` contains **physical** block ids; it may be extended during
      shadow decode when the logical sequence grows.
    """

    block_table: list[int]
    num_computed: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    sampling_params: ShadowSamplingParams


@dataclass(slots=True)
class ShadowPagedAttentionBatch:
    """Batch metadata for CPU paged attention (unified decode + prefill).

    This is the compact "kernel-facing" view built from per-request state.

    - ``query_token_ids``: one list per scheduled request. The inner list length
      is the query length for that request row (see
      :meth:`ShadowKvState.build_attention_batch`). Decode is the special case
      where each inner list has length 1.
    - ``kv_caches``: list of per-layer KV tensors. Each tensor is the shadow CPU
      paged cache for one transformer layer.
    - ``block_table``: int32 tensor of shape ``[B, max_blocks]`` containing
      per-request physical block ids, padded with zeros past each request's
      current block-table length.
    - ``slot_mapping``: int64 tensor of length ``sum(query_lens)``. For each
      query token (flattened in request order), maps to a *slot index* into the
      paged cache address space: ``phys_block * block_size + (pos % block_size)``.
    - ``seq_lens``: int32 tensor of length ``B`` giving the **post-query**
      sequence length (i.e. ``num_computed + query_len``) per request row.
      The CPU kernel indexes KV length and block tables **per request**, not
      per token.
    - ``query_start_loc``: int32 prefix sum of query lengths of shape ``[B+1]``
      (classic flattened-batch segment offsets).
    - ``positions``: int64 tensor of length ``sum(query_lens)``. For each query
      token, the absolute token position within the request sequence at batch
      build time: ``num_computed .. num_computed + query_len - 1``.
    - ``max_seq_len``: max of ``seq_lens`` (post-query) across the batch.
    """

    query_token_ids: list[list[int]]
    kv_caches: list[torch.Tensor]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    positions: torch.Tensor
    max_seq_len: int


class ShadowKvState:
    """Owns the shadow CPU paged KV cache and per-request sequence metadata.

    Responsibilities
    ----------------
    - **Ingest migrated KV blocks**: each transformer layer registers exactly one
      CPU tensor (typically backed by memfd) containing the full paged cache for
      that layer.
    - **Track per-request state**: token ids, ``num_computed`` (KV-written
      length), and the logical→physical block table.
    - **Allocate blocks during growth**: if shadow decode extends a sequence
      beyond the migrated KV range, additional physical blocks are assigned from
      the local free pool.
    - **Build kernel-facing batch metadata** for unified prefill/decode CPU
      paged attention.

    This type deliberately avoids engine/scheduler imports so the shadow
    process can become ready quickly.
    """

    __slots__ = (
        "_num_layers",
        "_num_blocks",
        "_block_size",
        "_expected_dtype",
        "_expected_shape",
        "_requests",
        "_layers",
        "_free_blocks",
    )

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: str,
    ) -> None:
        self._num_layers = num_layers
        self._num_blocks = num_blocks
        self._block_size = block_size

        self._expected_dtype = torch_dtype_from_str(dtype)
        self._expected_shape = (
            2,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self._requests: dict[str, Request] = {}
        self._layers: list[MemfdTensor | None] = [None] * num_layers
        self._free_blocks: set[int] = set(range(num_blocks))

        logger.debug(
            "ShadowKvState num_blocks=%s free_blocks=%d",
            num_blocks,
            len(self._free_blocks),
        )

    def register_layer(
        self,
        layer_idx: int,
        tensor: MemfdTensor,
    ) -> None:
        if layer_idx < 0 or layer_idx >= self._num_layers:
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

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
        num_computed_tokens: int,
        block_table: list[int],
        sampling_params: ShadowSamplingParams,
    ) -> None:
        # Validate arguments before adding to the manager.
        if request_id in self._requests:
            raise ValueError(f"request_id already exists: {request_id!r}")
        for bid in block_table:
            if bid not in self._free_blocks:
                raise ValueError(
                    f"request {request_id!r}: block {bid} not in free blocks"
                )
        prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else []
        # Validate that the num_computed_tokens is not greater than
        # the total number of tokens.
        if num_computed_tokens > len(prompt_token_ids) + len(output_token_ids):
            raise ValueError(
                f"request {request_id!r}: num_computed_tokens={num_computed_tokens} "
                f"> len(prompt_token_ids)={len(prompt_token_ids)} + "
                f"len(output_token_ids)={len(output_token_ids)}"
            )
        # Validate that when output_token_ids is not empty,
        # all tokens are computed except the last one.
        if (
            output_token_ids
            and num_computed_tokens != len(prompt_token_ids) + len(output_token_ids) - 1
        ):
            raise ValueError(
                f"request {request_id!r}: num_computed_tokens={num_computed_tokens} "
                f"!= len(prompt_token_ids)={len(prompt_token_ids)} + "
                f"len(output_token_ids)={len(output_token_ids)} - 1"
            )
        # Add the request to the manager and update the free blocks.
        self._requests[request_id] = Request(
            block_table=block_table,
            num_computed=num_computed_tokens,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            sampling_params=sampling_params,
        )
        for bid in block_table:
            self._free_blocks.discard(bid)

    @property
    def requests(self) -> dict[str, Request]:
        return self._requests

    @property
    def layers(self) -> list[MemfdTensor | None]:
        return self._layers

    def has_requests(self) -> bool:
        return bool(self._requests)

    def has_free_blocks(self) -> bool:
        return bool(self._free_blocks)

    def request_ids(self) -> list[str]:
        return list(self._requests.keys())

    def prepare_request(
        self, request_id: str
    ) -> tuple[list[int], int, ShadowSamplingParams, bool]:
        """Prepare a request for forward pass.

        Returns:
        - uncomputed token ids
        - number of computed tokens
        - sampling parameters
        - block allocation success
        """
        if request_id not in self._requests:
            raise ValueError(f"unknown request_id: {request_id!r}")
        req = self._requests[request_id]
        prompt = req.prompt_token_ids
        output = req.output_token_ids
        nc = req.num_computed
        table = req.block_table
        p_len = len(prompt)
        # Get the uncomputed token ids.
        tids = prompt[nc:] + output if nc < p_len else output[nc - p_len :]
        # Ensure blocks exist for the uncomputed token ids.
        for pos in range(nc, nc + len(tids)):
            logical_block_idx = pos // self._block_size
            successed = self._ensure_block_for_logical(table, logical_block_idx)
            if not successed:
                return tids, nc, req.sampling_params, False
        return tids, nc, req.sampling_params, True

    def _ensure_block_for_logical(
        self, block_table: list[int], logical_block_idx: int
    ) -> bool:
        """Ensure a block exists for the logical block index.

        Returns True if the block exists, False otherwise.
        """
        while logical_block_idx >= len(block_table):
            if not self._free_blocks:
                return False
            block_table.append(self._free_blocks.pop())
        return True

    def advance_decoded_token(
        self,
        request_id: str,
        token_id: int,
    ) -> None:
        """Advance KV-written length and append one decoded token."""
        if request_id not in self._requests:
            raise ValueError(f"unknown request_id: {request_id!r}")
        req = self._requests[request_id]
        req.num_computed = len(req.prompt_token_ids) + len(req.output_token_ids)
        req.output_token_ids.append(token_id)

    def free_request(self, request_id: str) -> None:
        """Remove a request and release all its KV blocks."""
        if request_id not in self._requests:
            raise ValueError(f"unknown request_id: {request_id!r}")
        req = self._requests.pop(request_id)
        # Return all allocated KV blocks to the free pool
        self._free_blocks.update(req.block_table)

    def build_attention_batch(
        self,
        request_ids: list[str],
        query_token_ids: list[list[int]],
    ) -> ShadowPagedAttentionBatch:
        """Build the kernel-facing batch view from per-request queries.

        ``query_token_ids`` provides the per-request query tokens to be processed
        next (either prefill chunks or decode tokens). Positions and slot mapping
        are derived from current ``num_computed`` and the request's block table.
        """
        if not request_ids:
            raise ValueError("empty batch")
        if len(query_token_ids) != len(request_ids):
            raise ValueError("query_token_ids and request_ids mismatch")

        query_lens = [len(ch) for ch in query_token_ids]
        if any(L <= 0 for L in query_lens):
            raise ValueError("each query_token_ids chunk must be non-empty")

        block_size = int(self._block_size)

        n = len(request_ids)
        max_blocks = max(len(self._requests[rid].block_table) for rid in request_ids)
        block_table = torch.zeros((n, max_blocks), dtype=torch.int32)

        total_tokens = sum(query_lens)
        slot_mapping = torch.empty((total_tokens,), dtype=torch.int64)
        positions = torch.empty((total_tokens,), dtype=torch.int64)
        seq_lens = torch.empty((n,), dtype=torch.int32)
        qsl = torch.empty((n + 1,), dtype=torch.int32)

        sm = 0
        cum = 0
        qsl[0] = 0
        for row, (rid, L) in enumerate(zip(request_ids, query_lens, strict=True)):
            rq = self._requests[rid]
            blks = rq.block_table
            nb = len(blks)
            if nb > max_blocks:
                raise RuntimeError("internal: max_blocks too small")
            block_table[row, :nb] = torch.tensor(blks, dtype=torch.int32)
            start = int(rq.num_computed)
            seq_lens[row] = start + L
            cum += L
            qsl[row + 1] = cum
            positions[sm : sm + L] = torch.arange(
                start, start + L, dtype=torch.int64, device=block_table.device
            )
            for j in range(L):
                pos = start + j
                logical_block = pos // block_size
                if logical_block >= len(blks):
                    raise RuntimeError(
                        "block table too short for position "
                        f"(request_id={rid!r} pos={pos} logical_block={logical_block})"
                    )
                phys = int(blks[logical_block])
                slot_mapping[sm + j] = phys * block_size + (pos % block_size)
            sm += L
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

    def kv_cache_usage(self) -> float:
        """Calculate the KV cache usage."""
        return (self._num_blocks - len(self._free_blocks)) / self._num_blocks
