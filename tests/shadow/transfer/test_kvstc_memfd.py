# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.shadow.transfer.kvstc_memfd`` (no KVHTS imports on wire path)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import pytest
import torch

from vllm.shadow.transfer.kv_transport_common import dtype_str_from_torch
from vllm.shadow.transfer.kvstc_memfd import (
    MemfdTensor,
    UdsMemfdKvstcReceiverTransport,
    UdsMemfdKvstcSenderTransport,
)
from vllm.shadow.transfer.kvstc_protocol import KvstcHandoff, KvstcRequest


@pytest.fixture
def kvstc_ipc_path(tmp_path) -> str:
    return str(tmp_path / "kvstc.sock")


def _sample_requests() -> list[KvstcRequest]:
    """KVSTC wire uses :class:`KvstcRequest` (not KVHTS request shape)."""
    return [
        KvstcRequest(
            request_id="req-stc-1",
            token_ids=[10, 11, 12, 20, 21],
            block_table=[0, 1],
        ),
        KvstcRequest(
            request_id="req-stc-2",
            token_ids=[],
            block_table=[2],
        ),
    ]


def _rand_kv(
    shadow_num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        2, shadow_num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
    )


def _threaded_uds(
    ipc_path: str,
    run_receiver: Callable[[UdsMemfdKvstcReceiverTransport], None],
    run_sender: Callable[[UdsMemfdKvstcSenderTransport], None],
) -> None:
    err: list[BaseException] = []
    r = UdsMemfdKvstcReceiverTransport()
    r.prepare(ipc_path)
    s = UdsMemfdKvstcSenderTransport()

    def _r() -> None:
        try:
            r.accept_once()
            run_receiver(r)
        except BaseException as e:  # pragma: no cover
            err.append(e)
        finally:
            r.close()

    t = threading.Thread(target=_r)
    t.start()
    try:
        s.connect(ipc_path)
        run_sender(s)
    finally:
        s.close()
        t.join()
    if err:
        raise err[0]


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="memfd_create not available on this platform",
)
def test_kvstc_memfd_handoff_two_layers(kvstc_ipc_path: str) -> None:
    records = _sample_requests()
    migration_id = 42
    shadow_n, heads, bsz, hdim = 8, 2, 16, 8
    dtype = torch.float16
    handoff = KvstcHandoff(
        migration_id=migration_id,
        num_layers=2,
        batch_size=len(records),
        shadow_num_blocks=shadow_n,
        num_kv_heads=heads,
        head_dim=hdim,
        block_size=bsz,
        dtype=dtype_str_from_torch(dtype),
        requests=list(records),
    )
    p0 = _rand_kv(shadow_n, heads, bsz, hdim, dtype=dtype, seed=11)
    p1 = _rand_kv(shadow_n, heads, bsz, hdim, dtype=dtype, seed=22)
    for t in (p0, p1):
        for blk in range(shadow_n):
            t[:, blk].fill_(float(blk))

    got_ho: list[KvstcHandoff] = []
    got_layer_idx: list[int] = []

    def recv(rx: UdsMemfdKvstcReceiverTransport) -> None:
        ho = rx.recv_handoff()
        got_ho.append(ho)
        for li, want in ((0, p0), (1, p1)):
            mt = rx.recv_layer(ho)
            got_layer_idx.append(li)
            assert torch.allclose(mt.tensor, want)
            mt.close()

    def send(sx: UdsMemfdKvstcSenderTransport) -> None:
        sx.send_handoff(handoff)
        sx.send_layer(MemfdTensor.from_tensor(p0))
        sx.send_layer(MemfdTensor.from_tensor(p1))

    _threaded_uds(kvstc_ipc_path, recv, send)

    assert len(got_ho) == 1
    g = got_ho[0]
    assert g.migration_id == migration_id
    assert g.num_layers == 2
    assert g.batch_size == len(records)
    for i, r in enumerate(records):
        gr = g.requests[i]
        assert gr.request_id == r.request_id
        assert gr.token_ids == r.token_ids
        assert gr.block_table == r.block_table
    assert got_layer_idx == [0, 1]
