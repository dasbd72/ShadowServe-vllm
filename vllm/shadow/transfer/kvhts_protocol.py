# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTS: protocol for hot GPU → shadow CPU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

__all__ = (
    "KVHTS_HANDOFF_VERSION",
    "REQUIRED_KVHTS_HANDOFF_KEYS",
    "REQUIRED_KVHTS_REQUEST_KEYS",
    "KvhtsRequest",
    "KvhtsHandoff",
)

KVHTS_HANDOFF_VERSION: Final[int] = 1

REQUIRED_KVHTS_HANDOFF_KEYS: Final[frozenset[str]] = frozenset(
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
        "tksth_ipc_path",
    )
)

REQUIRED_KVHTS_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    (
        "request_id",
        "client_index",
        "prompt_token_ids",
        "output_token_ids",
        "num_computed_tokens",
        "block_table",
        "sampling_params",
    )
)


@dataclass(frozen=True, slots=True)
class KvhtsRequest:
    """Per-request chat-completion decode snapshot (no hot ``Request`` import)."""

    request_id: str
    client_index: int
    prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    num_computed_tokens: int
    block_table: list[int]
    sampling_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "client_index": self.client_index,
            "prompt_token_ids": self.prompt_token_ids,
            "output_token_ids": list(self.output_token_ids),
            "num_computed_tokens": self.num_computed_tokens,
            "block_table": list(self.block_table),
            "sampling_params": self.sampling_params,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> KvhtsRequest:
        missing = REQUIRED_KVHTS_REQUEST_KEYS - obj.keys()
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
        bt_raw = obj["block_table"]
        if not isinstance(bt_raw, list):
            raise ValueError("block_table must be a JSON array of integers")
        block_table = [int(x) for x in bt_raw]
        sp = obj["sampling_params"]
        if not isinstance(sp, dict):
            raise ValueError("sampling_params must be a JSON object")
        return KvhtsRequest(
            request_id=request_id,
            client_index=client_index,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            num_computed_tokens=num_computed_tokens,
            block_table=block_table,
            sampling_params=dict(sp),
        )


@dataclass(frozen=True, slots=True)
class KvhtsHandoff:
    """Full migration handoff payload as one class (chat completions only)."""

    migration_id: int
    num_layers: int
    batch_size: int
    shadow_num_blocks: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: str
    requests: list[KvhtsRequest]
    tksth_ipc_path: str
    """Hot TKSTH Unix socket path (non-empty)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": KVHTS_HANDOFF_VERSION,
            "migration_id": self.migration_id,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "shadow_num_blocks": self.shadow_num_blocks,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "requests": [r.to_dict() for r in self.requests],
            "tksth_ipc_path": self.tksth_ipc_path,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> KvhtsHandoff:
        missing = REQUIRED_KVHTS_HANDOFF_KEYS - obj.keys()
        if missing:
            raise ValueError(f"migration handoff missing keys: {sorted(missing)}")
        ver = obj.get("version")
        if ver != KVHTS_HANDOFF_VERSION:
            raise ValueError(
                f"unsupported migration handoff JSON version {ver!r}, "
                f"expected {KVHTS_HANDOFF_VERSION}"
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
        requests = [KvhtsRequest.from_dict(x) for x in reqs_raw]
        tksth_ipc_path_raw = obj["tksth_ipc_path"]
        if not isinstance(tksth_ipc_path_raw, str):
            raise ValueError("tksth_ipc_path must be a string")
        tksth_ipc_path = tksth_ipc_path_raw.strip()
        return KvhtsHandoff(
            migration_id=migration_id,
            num_layers=num_layers,
            batch_size=batch_size,
            shadow_num_blocks=shadow_num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            requests=requests,
            tksth_ipc_path=tksth_ipc_path,
        )

    def to_bytes(self) -> bytes:
        obj = self.to_dict()
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    @staticmethod
    def from_bytes(data: bytes) -> KvhtsHandoff:
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
        return KvhtsHandoff.from_dict(obj)
