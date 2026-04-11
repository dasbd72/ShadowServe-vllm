# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.shadow.transfer.kvhtc_memfd``.

KVHTC wire format (chat completions):

- **Handoff** (once per batch): framed bytes carrying migration JSON (flat layout
  fields + ``requests`` with ``sampling_params`` (e.g. ``max_tokens``),
  ``dst_block_table`` per request).
- **Layer batch** (per layer, after handoff): memfd FD passing via ``SCM_RIGHTS``.
  The inline message payload carries ``MigrationLayerEnvelope`` bytes; KV shape comes
  from the handoff's flat layout fields, not the layer memfd.
- Receiver ACKs after each message and (for layers) exposes a tensor view over the
  shared KV payload.
"""

from __future__ import annotations

import os
import threading

import pytest
import torch

from vllm.shadow.transfer.kvhtc_memfd import (
    UdsMemfdKvHtcReceiverTransport,
    UdsMemfdKvHtcSenderTransport,
)
from vllm.shadow.transfer.kvhtc_protocol import (
    HandoffRequest,
    MigrationHandoff,
    MigrationLayerEnvelope,
    dtype_str_from_torch,
)

# Dummy TKCTH path for wire tests (protocol requires a non-empty string).
_TEST_TKCTH_IPC_PATH = "/tmp/vllm-kvhtc-test-tkcth"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kvhtc_ipc_path(tmp_path) -> str:
    return str(tmp_path / "kvhtc.sock")


pytestmark = pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="memfd_create not available on this platform",
)


def _make_pinned(
    shadow_num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    # Avoid ``pin_memory=True`` here: it can force CUDA init and break CPU-only CI.
    return torch.randn(
        2,
        shadow_num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype=dtype,
    )


def _sample_handoff_requests() -> list[HandoffRequest]:
    return [
        HandoffRequest(
            request_id="req-handoff-1",
            client_index=0,
            prompt_token_ids=[10, 11, 12],
            output_token_ids=[20, 21],
            num_computed_tokens=5,
            num_prompt_tokens=3,
            dst_block_table=[0, 1],
            sampling_params={"temperature": 0.7, "max_tokens": 128},
        ),
        HandoffRequest(
            request_id="req-handoff-2",
            client_index=1,
            prompt_token_ids=None,
            output_token_ids=[],
            num_computed_tokens=0,
            num_prompt_tokens=1,
            dst_block_table=[2],
            sampling_params={"temperature": 1.0, "max_tokens": 256},
        ),
    ]


def _migration_handoff_for_records(
    records: list[HandoffRequest],
    *,
    num_layers: int = 4,
    shadow_num_blocks: int = 8,
    num_kv_heads: int = 2,
    block_size: int = 16,
    head_dim: int = 8,
    dtype: str = "float16",
    tkcth_ipc_path: str = _TEST_TKCTH_IPC_PATH,
) -> MigrationHandoff:
    batch_size = len(records)
    return MigrationHandoff(
        migration_id=7,
        num_layers=num_layers,
        batch_size=batch_size,
        shadow_num_blocks=shadow_num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        requests=list(records),
        tkcth_ipc_path=tkcth_ipc_path,
    )


def _handoff_round_trip_once(
    ipc_path: str,
    *,
    migration_id: int = 7,
    handoff: MigrationHandoff | None = None,
) -> MigrationHandoff:
    """Run handoff sender/receiver round-trip using a background receiver thread."""
    if handoff is None:
        handoff = _migration_handoff_for_records(_sample_handoff_requests())
    if handoff.migration_id != migration_id:
        handoff = MigrationHandoff(
            migration_id=migration_id,
            num_layers=handoff.num_layers,
            batch_size=handoff.batch_size,
            shadow_num_blocks=handoff.shadow_num_blocks,
            num_kv_heads=handoff.num_kv_heads,
            head_dim=handoff.head_dim,
            block_size=handoff.block_size,
            dtype=handoff.dtype,
            requests=handoff.requests,
            tkcth_ipc_path=handoff.tkcth_ipc_path,
        )
    receiver = UdsMemfdKvHtcReceiverTransport()
    receiver.prepare(ipc_path)
    sender = UdsMemfdKvHtcSenderTransport()
    recv_result: list[MigrationHandoff] = []
    recv_err: list[BaseException] = []

    def _receiver() -> None:
        try:
            receiver.accept_once()
            got = receiver.recv_handoff()
            recv_result.append(got)
        except BaseException as e:  # pragma: no cover
            recv_err.append(e)
        finally:
            receiver.close()

    t = threading.Thread(target=_receiver)
    t.start()
    try:
        sender.connect(ipc_path)
        sender.send_handoff(handoff)
    finally:
        t.join()

    if recv_err:
        raise recv_err[0]
    assert len(recv_result) == 1
    return recv_result[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migration_handoff_round_trip(kvhtc_ipc_path: str) -> None:
    """Handoff memfd: envelope fields and migration handoff match after JSON parse."""
    records = _sample_handoff_requests()
    handoff = _migration_handoff_for_records(records)
    migration_id = 99
    got = _handoff_round_trip_once(
        kvhtc_ipc_path, migration_id=migration_id, handoff=handoff
    )
    assert got.migration_id == migration_id
    assert got.num_layers == handoff.num_layers
    assert got.batch_size == handoff.batch_size
    assert len(got.requests) == len(records)
    for i, r in enumerate(records):
        assert got.requests[i].request_id == r.request_id
        assert got.requests[i].client_index == r.client_index
        assert got.requests[i].output_token_ids == r.output_token_ids
        assert got.requests[i].dst_block_table == r.dst_block_table
        assert got.requests[i].sampling_params == r.sampling_params
    assert got.tkcth_ipc_path == handoff.tkcth_ipc_path


def test_handoff_then_two_layer_batches_round_trip(kvhtc_ipc_path: str) -> None:
    """Full sequence: handoff → layer 0 → layer 1 (each with its own ACK)."""
    handoff_records = [
        HandoffRequest(
            request_id="solo",
            client_index=0,
            prompt_token_ids=[1],
            output_token_ids=[],
            num_computed_tokens=1,
            num_prompt_tokens=1,
            dst_block_table=[0],
            sampling_params={"max_tokens": 64},
        ),
    ]
    migration_id = 1001

    shadow_num_blocks = 3
    num_kv_heads = 1
    block_size = 4
    head_dim = 8
    dtype = torch.float16

    handoff = MigrationHandoff(
        migration_id=migration_id,
        num_layers=2,
        batch_size=1,
        shadow_num_blocks=shadow_num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype_str_from_torch(dtype),
        requests=list(handoff_records),
        tkcth_ipc_path=_TEST_TKCTH_IPC_PATH,
    )

    pinned0 = _make_pinned(
        shadow_num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, seed=1
    )
    pinned1 = _make_pinned(
        shadow_num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, seed=2
    )

    recv_err: list[BaseException] = []
    got_env: list = []
    got_layers: list = []
    receiver = UdsMemfdKvHtcReceiverTransport()
    receiver.prepare(kvhtc_ipc_path)
    sender = UdsMemfdKvHtcSenderTransport()

    def _receiver() -> None:
        try:
            receiver.accept_once()
            ho = receiver.recv_handoff()
            got_env.append(ho)
            for layer_idx, pinned in enumerate((pinned0, pinned1)):
                env_l, tensor = receiver.recv_layer(ho)
                got_layers.append((layer_idx, env_l, tensor))
        except BaseException as e:  # pragma: no cover
            recv_err.append(e)
        finally:
            receiver.close()

    t = threading.Thread(target=_receiver)
    t.start()
    try:
        sender.connect(kvhtc_ipc_path)
        sender.send_handoff(handoff)
        sender.send_layer(
            handoff,
            MigrationLayerEnvelope(layer_idx=0),
            pinned0,
        )
        sender.send_layer(
            handoff,
            MigrationLayerEnvelope(layer_idx=1),
            pinned1,
        )
    finally:
        t.join()

    if recv_err:
        raise recv_err[0]

    ho = got_env[0]
    assert ho.migration_id == migration_id
    assert len(ho.requests) == 1
    assert ho.requests[0].dst_block_table == [0]

    assert len(got_layers) == 2
    for layer_idx, env_l, tensor in got_layers:
        assert env_l.layer_idx == layer_idx
        assert torch.allclose(tensor.tensor, pinned0 if layer_idx == 0 else pinned1)
        tensor.close()


def test_multi_request_handoff_then_layer_uses_dst_from_handoff(
    kvhtc_ipc_path: str,
) -> None:
    """Handoff carries ``dst_block_table``; layer memfd has no per-slot JSON."""
    shadow_num_blocks = 6
    num_kv_heads = 1
    block_size = 2
    head_dim = 3
    dtype = torch.float32

    records = [
        HandoffRequest(
            request_id="req-A",
            client_index=0,
            prompt_token_ids=[1, 2],
            output_token_ids=[],
            num_computed_tokens=2,
            num_prompt_tokens=2,
            dst_block_table=[0, 2, 4],
            sampling_params={"max_tokens": 128},
        ),
        HandoffRequest(
            request_id="req-B",
            client_index=0,
            prompt_token_ids=[3],
            output_token_ids=[],
            num_computed_tokens=1,
            num_prompt_tokens=1,
            dst_block_table=[1, 3],
            sampling_params={"max_tokens": 128},
        ),
    ]

    handoff = MigrationHandoff(
        migration_id=42,
        num_layers=1,
        batch_size=2,
        shadow_num_blocks=shadow_num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype_str_from_torch(dtype),
        requests=list(records),
        tkcth_ipc_path=_TEST_TKCTH_IPC_PATH,
    )

    pinned = _make_pinned(
        shadow_num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype=dtype,
        seed=0,
    )
    for blk in range(shadow_num_blocks):
        pinned[:, blk].fill_(float(blk))

    recv_err: list[BaseException] = []
    got: list = []
    receiver = UdsMemfdKvHtcReceiverTransport()
    receiver.prepare(kvhtc_ipc_path)
    sender = UdsMemfdKvHtcSenderTransport()

    def _receiver() -> None:
        try:
            receiver.accept_once()
            ho = receiver.recv_handoff()
            env, tensor = receiver.recv_layer(ho)
            got.append((ho, env, tensor))
        except BaseException as e:  # pragma: no cover
            recv_err.append(e)
        finally:
            receiver.close()

    t = threading.Thread(target=_receiver)
    t.start()
    try:
        sender.connect(kvhtc_ipc_path)
        sender.send_handoff(handoff)
        sender.send_layer(
            handoff,
            MigrationLayerEnvelope(layer_idx=0),
            pinned,
        )
    finally:
        t.join()

    if recv_err:
        raise recv_err[0]

    ho, env, tensor = got[0]
    assert ho.batch_size == 2
    assert ho.requests[0].dst_block_table == [0, 2, 4]
    assert ho.requests[1].dst_block_table == [1, 3]

    for si, rec in enumerate(ho.requests):
        for blk in rec.dst_block_table:
            assert 0 <= blk < shadow_num_blocks
            slice_view = tensor.tensor[:, blk, :, :, :]
            assert torch.all(slice_view == float(blk)), (
                f"KV mismatch for handoff request index {si}, dst_block={blk}"
            )

    tensor.close()
