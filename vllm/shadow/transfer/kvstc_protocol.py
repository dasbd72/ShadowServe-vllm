# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVSTC: protocol for shadow CPU → cold GPU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

__all__ = (
    "KVSTC_HANDOFF_VERSION",
    "REQUIRED_KVSTC_HANDOFF_KEYS",
    "REQUIRED_KVSTC_REQUEST_KEYS",
    "KvstcRequest",
    "KvstcHandoff",
)

KVSTC_HANDOFF_VERSION: Final[int] = 1

REQUIRED_KVSTC_HANDOFF_KEYS: Final[frozenset[str]] = frozenset(
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
    )
)


REQUIRED_KVSTC_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    (
        "request_id",
        "token_ids",
        "block_table",
    )
)


@dataclass(frozen=True, slots=True)
class KvstcRequest:
    """Per-request KV state for shadow CPU → cold GPU."""

    request_id: str
    token_ids: list[int]
    block_table: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_ids": self.token_ids,
            "block_table": self.block_table,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> KvstcRequest:
        missing = REQUIRED_KVSTC_REQUEST_KEYS - obj.keys()
        if missing:
            raise ValueError(f"KVSTC request missing keys: {sorted(missing)}")
        request_id = str(obj["request_id"])
        token_ids = [int(x) for x in obj["token_ids"]]
        block_table = [int(x) for x in obj["block_table"]]
        return KvstcRequest(
            request_id=request_id,
            token_ids=token_ids,
            block_table=block_table,
        )


@dataclass(frozen=True, slots=True)
class KvstcHandoff:
    """Shadow→cold-GPU migration handoff."""

    migration_id: int
    num_layers: int
    batch_size: int
    shadow_num_blocks: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: str
    requests: list[KvstcRequest]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": KVSTC_HANDOFF_VERSION,
            "migration_id": self.migration_id,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "shadow_num_blocks": self.shadow_num_blocks,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "requests": [r.to_dict() for r in self.requests],
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> KvstcHandoff:
        missing = REQUIRED_KVSTC_HANDOFF_KEYS - obj.keys()
        if missing:
            raise ValueError(f"KVSTC handoff missing keys: {sorted(missing)}")
        ver = obj.get("version")
        if ver != KVSTC_HANDOFF_VERSION:
            raise ValueError(
                f"unsupported KVSTC handoff JSON version {ver!r}, "
                f"expected {KVSTC_HANDOFF_VERSION}"
            )
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
            raise ValueError("KVSTC handoff must contain a 'requests' array")
        if batch_size != len(reqs_raw):
            raise ValueError(
                f"batch_size {batch_size} != len(requests) {len(reqs_raw)}"
            )
        if not all(isinstance(x, dict) for x in reqs_raw):
            raise ValueError("each request must be a JSON object")
        requests = [KvstcRequest.from_dict(x) for x in reqs_raw]
        return KvstcHandoff(
            migration_id=migration_id,
            num_layers=num_layers,
            batch_size=batch_size,
            shadow_num_blocks=shadow_num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            requests=requests,
        )

    def to_bytes(self) -> bytes:
        obj = self.to_dict()
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    @staticmethod
    def from_bytes(data: bytes) -> KvstcHandoff:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("KVSTC handoff is not valid UTF-8") from e
        try:
            obj: Any = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("KVSTC handoff is not valid JSON") from e
        if not isinstance(obj, dict):
            raise ValueError("KVSTC handoff JSON must be an object")
        return KvstcHandoff.from_dict(obj)
