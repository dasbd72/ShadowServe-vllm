# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.shadow.transfer.kvhts_memfd``."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import pytest
import torch

from vllm.shadow.transfer.kv_transport_common import dtype_str_from_torch
from vllm.shadow.transfer.kvhts_memfd import (
    MemfdTensor,
    UdsMemfdKvhtsReceiverTransport,
    UdsMemfdKvhtsSenderTransport,
)
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff, KvhtsRequest

_TEST_TKSTH = "/tmp/vllm-kvhts-test-tksth"


@pytest.fixture
def kvhts_ipc_path(tmp_path) -> str:
    return str(tmp_path / "kvhts.sock")


def _sample_handoff_requests() -> list[KvhtsRequest]:
    return [
        KvhtsRequest(
            request_id="req-handoff-1",
            prompt_token_ids=[10, 11, 12],
            output_token_ids=[20, 21],
            num_computed_tokens=5,
            block_table=[0, 1],
            sampling_params={"temperature": 0.7, "max_tokens": 128},
        ),
        KvhtsRequest(
            request_id="req-handoff-2",
            prompt_token_ids=None,
            output_token_ids=[],
            num_computed_tokens=0,
            block_table=[2],
            sampling_params={"temperature": 1.0, "max_tokens": 256},
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
    run_receiver: Callable[[UdsMemfdKvhtsReceiverTransport], None],
    run_sender: Callable[[UdsMemfdKvhtsSenderTransport], None],
) -> None:
    err: list[BaseException] = []
    r = UdsMemfdKvhtsReceiverTransport()
    r.prepare(ipc_path)
    s = UdsMemfdKvhtsSenderTransport()

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
def test_memfd_handoff_two_layers_and_dst_block_views(kvhts_ipc_path: str) -> None:
    """Handoff round-trip, two layer memfds, multi-request handoff + dst table slice."""
    records = _sample_handoff_requests()
    migration_id = 99
    shadow_n, heads, bsz, hdim = 8, 2, 16, 8
    dtype = torch.float16
    handoff = KvhtsHandoff(
        migration_id=migration_id,
        num_layers=2,
        batch_size=len(records),
        shadow_num_blocks=shadow_n,
        num_kv_heads=heads,
        head_dim=hdim,
        block_size=bsz,
        dtype=dtype_str_from_torch(dtype),
        requests=list(records),
        tksth_ipc_path=_TEST_TKSTH,
    )
    p0 = _rand_kv(shadow_n, heads, bsz, hdim, dtype=dtype, seed=1)
    p1 = _rand_kv(shadow_n, heads, bsz, hdim, dtype=dtype, seed=2)
    for t in (p0, p1):
        for blk in range(shadow_n):
            t[:, blk].fill_(float(blk))

    got_ho: list[KvhtsHandoff] = []
    got_layer_idx: list[int] = []

    def recv(rx: UdsMemfdKvhtsReceiverTransport) -> None:
        ho = rx.recv_handoff()
        got_ho.append(ho)
        for li, want in ((0, p0), (1, p1)):
            mt = rx.recv_layer(ho)
            got_layer_idx.append(li)
            assert torch.allclose(mt.tensor, want)
            if li == 0:
                for rec in ho.requests:
                    for blk in rec.block_table:
                        sl = mt.tensor[:, blk, :, :, :]
                        assert torch.all(sl == float(blk)), (rec.request_id, blk)
            mt.close()

    def send(sx: UdsMemfdKvhtsSenderTransport) -> None:
        sx.send_handoff(handoff)
        sx.send_layer(MemfdTensor.from_tensor(p0))
        sx.send_layer(MemfdTensor.from_tensor(p1))

    _threaded_uds(kvhts_ipc_path, recv, send)

    assert len(got_ho) == 1
    g = got_ho[0]
    assert g.migration_id == migration_id
    assert g.num_layers == 2
    assert g.batch_size == len(records)
    assert g.tksth_ipc_path == _TEST_TKSTH
    for i, r in enumerate(records):
        gr = g.requests[i]
        assert gr.request_id == r.request_id
        assert gr.output_token_ids == r.output_token_ids
        assert gr.block_table == r.block_table
        assert gr.sampling_params == r.sampling_params
    assert got_layer_idx == [0, 1]
