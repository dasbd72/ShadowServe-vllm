# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hot GPU worker: gather KV for shadow migration and send via KVHTC transport."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm._custom_ops import gather_kv_blocks_batched
from vllm.logger import init_logger
from vllm.shadow.transfer.kvhtc_memfd import UdsMemfdKvHtcSenderTransport
from vllm.shadow.transfer.kvhtc_protocol import (
    HandoffRequest,
    MigrationHandoff,
    MigrationLayerEnvelope,
    dtype_str_from_torch,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _msgspec_to_dict(obj: object | None) -> dict[str, Any] | None:
    if obj is None:
        return None
    return msgspec.to_builtins(obj)


def _gather_src_block_tables(
    model_runner: GPUModelRunner,
    request_ids: list[str],
) -> tuple[list[list[int]], list[int]]:
    """Per-request GPU block ids (single KV group) and block counts."""
    src_block_tables: list[list[int]] = []
    num_blocks_per_req: list[int] = []

    for req_id in request_ids:
        req_state = model_runner.requests.get(req_id)
        if req_state is None:
            raise RuntimeError(
                f"kvhtc migration: request {req_id!r} not in model runner cache"
            )
        if len(req_state.block_ids) != 1:
            raise NotImplementedError(
                "kvhtc migration currently supports a single KV cache group; "
                f"got {len(req_state.block_ids)} for {req_id!r}"
            )
        gpu_blocks = list(req_state.block_ids[0])
        if not gpu_blocks:
            raise RuntimeError(
                f"kvhtc migration: request {req_id!r} has no allocated KV blocks"
            )
        src_block_tables.append(gpu_blocks)
        num_blocks_per_req.append(len(gpu_blocks))

    return src_block_tables, num_blocks_per_req


def _allocate_dst_block_tables(
    num_blocks_per_req: list[int],
    additional_per_request: int,
) -> tuple[list[list[int]], int]:
    """Shadow-side slot indices packed consecutively across the batch."""
    dst_block_tables: list[list[int]] = []
    offset = 0
    for n in num_blocks_per_req:
        dst_block_tables.append(list(range(offset, offset + n)))
        offset += n
    return dst_block_tables, offset + additional_per_request * len(num_blocks_per_req)


def _handoff_requests(
    model_runner: GPUModelRunner,
    request_ids: list[str],
    client_indices: list[int],
    dst_block_tables: list[list[int]],
) -> list[HandoffRequest]:
    rows: list[HandoffRequest] = []
    for i, req_id in enumerate(request_ids):
        req_state = model_runner.requests[req_id]
        sp = req_state.sampling_params
        pp = req_state.pooling_params
        if pp is not None:
            raise NotImplementedError(
                "kvhtc migration supports chat completions only; "
                f"request {req_id!r} has pooling_params set"
            )
        if sp is None:
            raise NotImplementedError(
                "kvhtc migration requires sampling_params; "
                f"request {req_id!r} has no sampling_params"
            )
        sampling_wire = _msgspec_to_dict(sp)
        if not isinstance(sampling_wire, dict):
            raise RuntimeError(
                f"kvhtc migration: sampling_params must serialize to a dict "
                f"for {req_id!r}"
            )
        num_computed_tokens = int(req_state.num_tokens - 1)
        if num_computed_tokens <= 0:
            raise RuntimeError(
                f"kvhtc migration: request {req_id!r} has no computed tokens "
                f"(num_tokens={req_state.num_tokens})"
            )
        rows.append(
            HandoffRequest(
                request_id=req_id,
                client_index=int(client_indices[i]),
                prompt_token_ids=(
                    list(req_state.prompt_token_ids)
                    if req_state.prompt_token_ids is not None
                    else None
                ),
                output_token_ids=req_state.output_token_ids,
                num_computed_tokens=num_computed_tokens,
                num_prompt_tokens=int(req_state.num_prompt_tokens),
                dst_block_table=dst_block_tables[i],
                sampling_params=sampling_wire,
            )
        )
    return rows


def _remove_requests_from_runner(
    model_runner: GPUModelRunner, request_ids: list[str]
) -> None:
    for req_id in request_ids:
        model_runner.requests.pop(req_id, None)
        model_runner.num_prompt_logprobs.pop(req_id, None)
        model_runner.input_batch.remove_request(req_id)


def execute_shadow_kv_migration_for_model_runner(
    model_runner: GPUModelRunner,
    kvhtc_ipc_path: str,
    migration_id: int,
    request_ids: list[str],
    client_indices: list[int],
    tkcth_ipc_path: str,
) -> None:
    """Gather KV layer-by-layer, send handoff + layers over ``kvhtc_ipc_path``.

    Removes migrated requests from the model runner persistent batch and cache.
    """
    if len(request_ids) != len(client_indices):
        raise ValueError("request_ids and client_indices length mismatch")
    if not request_ids:
        return

    kv_caches: list[torch.Tensor] | None = getattr(model_runner, "kv_caches", None)
    if not kv_caches:
        raise RuntimeError("kvhtc migration requires a non-empty kv_caches list")

    src_block_tables, num_blocks_per_req = _gather_src_block_tables(
        model_runner, request_ids
    )
    shadow_migration_config = model_runner.vllm_config.shadow_migration_config
    additional = shadow_migration_config.shadow_additional_blocks_per_request
    dst_block_tables, shadow_num_blocks = _allocate_dst_block_tables(
        num_blocks_per_req, additional
    )

    if not str(tkcth_ipc_path).strip():
        raise ValueError("tkcth_ipc_path must be a non-empty string")

    kv0 = kv_caches[0]
    _, _, block_size, num_kv_heads, head_dim = map(int, kv0.shape)
    batch_size = len(request_ids)
    num_layers = len(kv_caches)
    handoff = MigrationHandoff(
        migration_id=int(migration_id),
        num_layers=int(num_layers),
        batch_size=int(batch_size),
        shadow_num_blocks=int(shadow_num_blocks),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        block_size=int(block_size),
        dtype=dtype_str_from_torch(kv0.dtype),
        requests=_handoff_requests(
            model_runner, request_ids, client_indices, dst_block_tables
        ),
        tkcth_ipc_path=tkcth_ipc_path,
    )

    t0 = time.perf_counter()
    transport = UdsMemfdKvHtcSenderTransport()
    try:
        transport.connect(kvhtc_ipc_path)
        transport.send_handoff(handoff)

        num_blocks_t = torch.tensor(num_blocks_per_req, dtype=torch.int32, device="cpu")

        pinned_kv = torch.empty(
            2,
            shadow_num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=kv0.dtype,
            device="cpu",
            pin_memory=True,
        )

        for layer_idx, kv_cache in enumerate(kv_caches):
            if kv_cache.shape != kv0.shape:
                raise NotImplementedError(
                    "kvhtc migration expects uniform KV shape per layer; "
                    f"layer 0 {tuple(kv0.shape)} vs layer {layer_idx} "
                    f"{tuple(kv_cache.shape)}"
                )
            gather_kv_blocks_batched(
                kv_cache,
                src_block_tables,
                pinned_kv,
                dst_block_tables,
                num_blocks_t,
            )
            torch.cuda.synchronize()
            transport.send_layer(
                handoff,
                MigrationLayerEnvelope(layer_idx=int(layer_idx)),
                pinned_kv,
            )

        _remove_requests_from_runner(model_runner, request_ids)

        elapsed = time.perf_counter() - t0
        total_bytes = num_layers * pinned_kv.numel() * pinned_kv.element_size()
        logger.info(
            "kvhtc migration batch done: kvhtc_migration_id=%s kvhtc_requests=%d "
            "kvhtc_layers=%d kvhtc_shadow_blocks=%d kvhtc_bytes≈%d kvhtc_time_s=%.3f "
            "kvhtc_tkcth_ipc_path=%s",
            migration_id,
            batch_size,
            num_layers,
            shadow_num_blocks,
            total_bytes,
            elapsed,
            tkcth_ipc_path,
        )
    finally:
        transport.close()
