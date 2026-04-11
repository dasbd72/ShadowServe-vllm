# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTC wire format: binary envelopes and migration JSON for hot GPU → cold shadow CPU.

Scope: **chat completions** (text generation with ``SamplingParams`` on the wire).
Pooling, embeddings, and multimodal inputs are out of scope and not represented.

Used by transports such as UDS + ``memfd`` + ``SCM_RIGHTS``; the layout is
transport-agnostic.

**Migration handoff** (sent once per batch before any KV layer):

JSON is versioned (see :data:`MIGRATION_HANDOFF_VERSION`) and includes both the KV
layout fields, the ``requests`` records (each with required ``sampling_params``), and
required ``tkcth_ipc_path`` (hot-side TKCTH Unix socket path for token streaming back
from the shadow).

**Layer batch** (one memfd per layer; ACK after each):

    [ MigrationLayerEnvelope ][ SharedLayerKV ]

``MigrationLayerEnvelope`` carries only wire version and ``layer_idx``.
The KV tensor is a contiguous byte image occupying the full memfd (the envelope is
carried in the inline UDS payload for ``SCM_RIGHTS``).

This module avoids importing vLLM. It imports ``torch`` only for
:class:`DTypeId` ↔ ``torch.dtype`` mapping used by memfd KV transfer.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Final

import torch

# -----------------------------------------------------------------------------
# Migration handoff
# -----------------------------------------------------------------------------

MIGRATION_HANDOFF_VERSION: Final[int] = 1

REQUIRED_MIGRATION_HANDOFF_KEYS: Final[frozenset[str]] = frozenset(
    (
        "migration_id",
        "num_layers",
        "batch_size",
        "shadow_num_blocks",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "dtype",
        "requests",
        "tkcth_ipc_path",
    )
)

REQUIRED_HANDOFF_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    (
        "request_id",
        "client_index",
        "prompt_token_ids",
        "output_token_ids",
        "num_computed_tokens",
        "num_prompt_tokens",
        "dst_block_table",
        "sampling_params",
    )
)


@dataclass(frozen=True, slots=True)
class HandoffRequest:
    """Per-request chat-completion decode snapshot (no hot ``Request`` import)."""

    request_id: str
    client_index: int
    prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    num_computed_tokens: int
    num_prompt_tokens: int
    dst_block_table: list[int]
    sampling_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "client_index": self.client_index,
            "prompt_token_ids": self.prompt_token_ids,
            "output_token_ids": list(self.output_token_ids),
            "num_computed_tokens": self.num_computed_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "dst_block_table": list(self.dst_block_table),
            "sampling_params": self.sampling_params,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> HandoffRequest:
        missing = REQUIRED_HANDOFF_REQUEST_KEYS - obj.keys()
        if missing:
            raise ValueError(f"handoff request missing keys: {sorted(missing)}")
        request_id = str(obj["request_id"])
        client_index = int(obj["client_index"])
        prompt_raw = obj["prompt_token_ids"]
        if prompt_raw is not None and not isinstance(prompt_raw, list):
            raise ValueError(
                "prompt_token_ids must be null or a JSON array of integers"
            )
        prompt_token_ids = None if prompt_raw is None else [int(x) for x in prompt_raw]
        out_raw = obj["output_token_ids"]
        if not isinstance(out_raw, list):
            raise ValueError("output_token_ids must be a JSON array of integers")
        output_token_ids = [int(x) for x in out_raw]
        num_computed_tokens = int(obj["num_computed_tokens"])
        num_prompt_tokens = int(obj["num_prompt_tokens"])
        dst_raw = obj["dst_block_table"]
        if not isinstance(dst_raw, list):
            raise ValueError("dst_block_table must be a JSON array of integers")
        dst_block_table = [int(x) for x in dst_raw]
        sp = obj["sampling_params"]
        if not isinstance(sp, dict):
            raise ValueError("sampling_params must be a JSON object")
        return HandoffRequest(
            request_id=request_id,
            client_index=client_index,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            num_computed_tokens=num_computed_tokens,
            num_prompt_tokens=num_prompt_tokens,
            dst_block_table=dst_block_table,
            sampling_params=dict(sp),
        )


@dataclass(frozen=True, slots=True)
class MigrationHandoff:
    """Full migration handoff payload as one class (chat completions only)."""

    migration_id: int
    num_layers: int
    batch_size: int
    shadow_num_blocks: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: str
    requests: list[HandoffRequest]
    tkcth_ipc_path: str
    """Hot TKCTH Unix socket path (non-empty)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MIGRATION_HANDOFF_VERSION,
            "migration_id": self.migration_id,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "shadow_num_blocks": self.shadow_num_blocks,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "requests": [r.to_dict() for r in self.requests],
            "tkcth_ipc_path": self.tkcth_ipc_path,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> MigrationHandoff:
        missing = REQUIRED_MIGRATION_HANDOFF_KEYS - obj.keys()
        if missing:
            raise ValueError(f"migration handoff missing keys: {sorted(missing)}")
        ver = obj.get("version")
        if ver != MIGRATION_HANDOFF_VERSION:
            raise ValueError(
                f"unsupported migration handoff JSON version {ver!r}, "
                f"expected {MIGRATION_HANDOFF_VERSION}"
            )
        if "migration_id" not in obj:
            raise ValueError("migration handoff must contain 'migration_id'")
        migration_id = int(obj["migration_id"])
        num_layers = int(obj["num_layers"])
        batch_size = int(obj["batch_size"])
        shadow_num_blocks = int(obj["shadow_num_blocks"])
        num_kv_heads = int(obj["num_kv_heads"])
        head_dim = int(obj["head_dim"])
        block_size = int(obj["block_size"])
        dtype = obj["dtype"]
        reqs_raw = obj["requests"]
        if not isinstance(reqs_raw, list):
            raise ValueError("migration handoff must contain a 'requests' array")
        if batch_size != len(reqs_raw):
            raise ValueError(
                f"batch_size {batch_size} != len(requests) {len(reqs_raw)}"
            )
        if not all(isinstance(x, dict) for x in reqs_raw):
            raise ValueError("each request must be a JSON object")
        requests = [HandoffRequest.from_dict(x) for x in reqs_raw]
        tkcth_ipc_path_raw = obj["tkcth_ipc_path"]
        if not isinstance(tkcth_ipc_path_raw, str):
            raise ValueError("tkcth_ipc_path must be a string")
        tkcth_ipc_path = tkcth_ipc_path_raw.strip()
        if not tkcth_ipc_path:
            raise ValueError("tkcth_ipc_path must be non-empty")
        return MigrationHandoff(
            migration_id=migration_id,
            num_layers=num_layers,
            batch_size=batch_size,
            shadow_num_blocks=shadow_num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            requests=requests,
            tkcth_ipc_path=tkcth_ipc_path,
        )

    def to_bytes(self) -> bytes:
        obj = self.to_dict()
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    @staticmethod
    def from_bytes(data: bytes) -> MigrationHandoff:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("migration handoff is not valid UTF-8") from e
        try:
            obj: Any = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("migration handoff is not valid JSON") from e
        if not isinstance(obj, dict):
            raise ValueError("migration handoff JSON must be an object")
        return MigrationHandoff.from_dict(obj)


# -----------------------------------------------------------------------------
# DType mapping helpers
# -----------------------------------------------------------------------------

# MigrationHandoff.dtype is a canonical, torch-free string (no "torch." prefix).
_DTYPE_STR_TO_TORCH: Final[dict[str, torch.dtype]] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def dtype_str_from_torch(dt: torch.dtype) -> str:
    for k, v in _DTYPE_STR_TO_TORCH.items():
        if v == dt:
            return k
    raise ValueError(f"unsupported torch dtype for KVHTC handoff: {dt}")


def torch_dtype_from_str(name: str) -> torch.dtype:
    if not isinstance(name, str):
        raise TypeError("dtype must be a string")
    key = name.strip()
    if key.startswith("torch."):
        key = key[len("torch.") :]
    dt = _DTYPE_STR_TO_TORCH.get(key)
    if dt is None:
        raise ValueError(f"unsupported dtype string for KVHTC handoff: {name!r}")
    return dt


# -----------------------------------------------------------------------------
# Binary envelope (fixed size, big-endian) — per-layer identity only
# -----------------------------------------------------------------------------

_BATCH_LAYER_ENVELOPE_MAGIC: Final[bytes] = b"VKLB"
_BATCH_LAYER_ENVELOPE_VERSION: Final[int] = 1

# magic + version (u32) + layer_idx (u32)
_BATCH_LAYER_ENVELOPE_STRUCT: Final[struct.Struct] = struct.Struct("!4sII")


@dataclass(frozen=True, slots=True)
class MigrationLayerEnvelope:
    """Fixed-size, big-endian record: wire version and which KV layer this memfd is."""

    layer_idx: int

    def to_bytes(self) -> bytes:
        return _BATCH_LAYER_ENVELOPE_STRUCT.pack(
            _BATCH_LAYER_ENVELOPE_MAGIC, _BATCH_LAYER_ENVELOPE_VERSION, self.layer_idx
        )

    @staticmethod
    def from_bytes(data: bytes) -> MigrationLayerEnvelope:
        if len(data) < _BATCH_LAYER_ENVELOPE_STRUCT.size:
            raise ValueError(
                f"need at least {_BATCH_LAYER_ENVELOPE_STRUCT.size} bytes "
                f"for envelope, got {len(data)}"
            )
        unpacked = _BATCH_LAYER_ENVELOPE_STRUCT.unpack_from(data, 0)
        got_magic = unpacked[0]
        if got_magic != _BATCH_LAYER_ENVELOPE_MAGIC:
            raise ValueError(
                f"bad envelope magic: {got_magic!r}, "
                f"expected {_BATCH_LAYER_ENVELOPE_MAGIC!r}"
            )
        _m, version, layer_idx = unpacked
        if version != _BATCH_LAYER_ENVELOPE_VERSION:
            raise ValueError(
                f"unsupported envelope version {version}, "
                f"expected {_BATCH_LAYER_ENVELOPE_VERSION}"
            )
        return MigrationLayerEnvelope(layer_idx=layer_idx)


def shared_kv_num_bytes(
    handoff: MigrationHandoff,
) -> int:
    """Number of bytes in `SharedLayerKV` for one layer (contiguous tensor image).

    Shape: ``(2, shadow_num_blocks, num_kv_heads, block_size, head_dim)``.
    """
    itemsize = torch_dtype_from_str(handoff.dtype).itemsize
    n = (
        2
        * handoff.shadow_num_blocks
        * handoff.num_kv_heads
        * handoff.block_size
        * handoff.head_dim
    )
    return n * itemsize
