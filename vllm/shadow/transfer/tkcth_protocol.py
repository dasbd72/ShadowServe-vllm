# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TKCTH wire protocol: framed JSON messages for chat-completion token streaming.

TKCTH is intentionally lightweight:
- stdlib-only (no vLLM imports)
- one JSON object per frame (framing is handled by :mod:`vllm.shadow.transfer.uds`)
- strict message validation by version + message type

Scope: **chat completions** (generated token deltas). Pooling/embeddings and other
tasks are not represented.

Message envelope fields (top-level JSON object):
- version: int (currently 1)
- type: str ("token_delta" | "finish" | "error")
- migration_id: int
- request_id: str (required for most message types; may be omitted for "error")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, TypeAlias

__all__ = (
    "TKCTH_MESSAGE_TYPES",
    "TkcthTokenDelta",
    "TkcthFinish",
    "TkcthError",
    "TkcthMessage",
)

TKCTH_MESSAGE_TYPES: Final[frozenset[str]] = frozenset(
    ("token_delta", "finish", "error")
)

_TKCTH_MESSAGE_VERSION: Final[int] = 1

_REQUIRED_COMMON: Final[frozenset[str]] = frozenset(("version", "type", "migration_id"))
_REQUIRED_COMMON_WITH_REQUEST: Final[frozenset[str]] = frozenset(
    ("version", "type", "migration_id", "request_id")
)

_REQUIRED_BY_TYPE: Final[dict[str, frozenset[str]]] = {
    "token_delta": frozenset(
        (
            "version",
            "type",
            "migration_id",
            "request_id",
            "token_ids",
        )
    ),
    "finish": _REQUIRED_COMMON_WITH_REQUEST,
    # For "error", request_id is optional (e.g. handshake-level failure).
    "error": frozenset(("version", "type", "migration_id", "message")),
}


class TkcthMessageBase:
    """Base class for all TKCTH messages.

    Subclasses must implement:
    - to_dict(self) -> dict[str, Any]
    - from_dict(obj: dict[str, Any]) -> <Subclass>

    This base class provides:
    - to_bytes(self) -> bytes: compact UTF-8 JSON encoding of to_dict()
    - from_bytes(cls, data: bytes) -> <Subclass>: JSON decode + cls.from_dict(...)
    """

    # Subclasses set these literals.
    type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("subclass must implement to_dict()")

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TkcthMessageBase:
        raise NotImplementedError("subclass must implement from_dict()")

    def to_bytes(self) -> bytes:
        obj = self.to_dict()
        TkcthMessageBase._validate_message_dict(obj)
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TkcthMessageBase:
        """Decode bytes into this message type via ``cls.from_dict``."""
        obj = TkcthMessageBase._decode_bytes_to_obj(data)
        if not isinstance(obj, dict):
            raise ValueError("TKCTH message JSON must be an object")
        TkcthMessageBase._validate_message_dict(obj)
        return cls.from_dict(obj)

    # ---- internal helpers (kept on the base; no free functions) ----

    @staticmethod
    def _decode_bytes_to_obj(data: bytes) -> Any:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("TKCTH message is not valid UTF-8") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("TKCTH message is not valid JSON") from e

    @staticmethod
    def _validate_message_dict(msg: dict[str, Any]) -> None:
        """Validate message shape and required fields (raises ValueError)."""
        missing_common = _REQUIRED_COMMON - msg.keys()
        if missing_common:
            raise ValueError(f"TKCTH message missing keys: {sorted(missing_common)}")

        typ = msg.get("type")
        if not isinstance(typ, str) or typ not in TKCTH_MESSAGE_TYPES:
            raise ValueError(
                f"unsupported TKCTH message type {typ!r}, expected one of "
                f"{sorted(TKCTH_MESSAGE_TYPES)}"
            )

        required = _REQUIRED_BY_TYPE[typ]
        missing = required - msg.keys()
        if missing:
            raise ValueError(f"TKCTH {typ} missing keys: {sorted(missing)}")

        migration_id = msg.get("migration_id")
        if not isinstance(migration_id, int):
            raise ValueError("migration_id must be an int")

        request_id = msg.get("request_id")
        if typ == "error":
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("request_id must be a str when present")
        else:
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty str")

        if typ == "token_delta":
            token_ids = msg.get("token_ids")
            if not isinstance(token_ids, list):
                raise ValueError("token_ids must be a JSON array of integers")
            for x in token_ids:
                if not isinstance(x, int):
                    raise ValueError("token_ids must contain only integers")

        elif typ == "finish":
            if (
                "finish_reason" in msg
                and msg["finish_reason"] is not None
                and not isinstance(msg["finish_reason"], str)
            ):
                raise ValueError("finish_reason must be a str or null")

        elif typ == "error":
            message = msg.get("message")
            if not isinstance(message, str) or not message:
                raise ValueError("message must be a non-empty str")
            if (
                "code" in msg
                and msg["code"] is not None
                and not isinstance(msg["code"], (int, str))
            ):
                raise ValueError("code must be an int, str, or null")


@dataclass(frozen=True, slots=True)
class TkcthTokenDelta(TkcthMessageBase):
    """Generated token ids for one chat-completion delta (no logprobs/text on wire)."""

    migration_id: int
    request_id: str
    token_ids: list[int]

    type: ClassVar[Literal["token_delta"]] = "token_delta"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _TKCTH_MESSAGE_VERSION,
            "type": "token_delta",
            "migration_id": int(self.migration_id),
            "request_id": str(self.request_id),
            "token_ids": [int(x) for x in self.token_ids],
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TkcthTokenDelta:
        if obj["version"] != _TKCTH_MESSAGE_VERSION:
            raise ValueError(
                f"unsupported TKCTH message version {obj['version']}, "
                f"expected {_TKCTH_MESSAGE_VERSION}"
            )
        if obj["type"] != "token_delta":
            raise ValueError("not a token_delta message")
        token_ids = obj["token_ids"]
        return cls(
            migration_id=int(obj["migration_id"]),
            request_id=str(obj["request_id"]),
            token_ids=[int(x) for x in token_ids],
        )


@dataclass(frozen=True, slots=True)
class TkcthFinish(TkcthMessageBase):
    migration_id: int
    request_id: str
    finish_reason: str | None = None

    type: ClassVar[Literal["finish"]] = "finish"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": _TKCTH_MESSAGE_VERSION,
            "type": "finish",
            "migration_id": int(self.migration_id),
            "request_id": str(self.request_id),
        }
        if self.finish_reason is not None:
            d["finish_reason"] = self.finish_reason
        return d

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TkcthFinish:
        if obj["version"] != _TKCTH_MESSAGE_VERSION:
            raise ValueError(
                f"unsupported TKCTH message version {obj['version']}, "
                f"expected {_TKCTH_MESSAGE_VERSION}"
            )
        if obj["type"] != "finish":
            raise ValueError("not a finish message")
        fr = obj.get("finish_reason")
        return cls(
            migration_id=int(obj["migration_id"]),
            request_id=str(obj["request_id"]),
            finish_reason=None if fr is None else str(fr),
        )


@dataclass(frozen=True, slots=True)
class TkcthError(TkcthMessageBase):
    migration_id: int
    message: str
    request_id: str | None = None
    code: int | str | None = None

    type: ClassVar[Literal["error"]] = "error"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": _TKCTH_MESSAGE_VERSION,
            "type": "error",
            "migration_id": int(self.migration_id),
            "message": str(self.message),
        }
        if self.request_id is not None:
            d["request_id"] = str(self.request_id)
        if self.code is not None:
            d["code"] = self.code
        return d

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TkcthError:
        if obj["version"] != _TKCTH_MESSAGE_VERSION:
            raise ValueError(
                f"unsupported TKCTH message version {obj['version']}, "
                f"expected {_TKCTH_MESSAGE_VERSION}"
            )
        if obj["type"] != "error":
            raise ValueError("not an error message")
        return cls(
            migration_id=int(obj["migration_id"]),
            message=str(obj["message"]),
            request_id=(
                None if obj.get("request_id") is None else str(obj.get("request_id"))
            ),
            code=obj.get("code"),
        )


TkcthMessage: TypeAlias = TkcthTokenDelta | TkcthFinish | TkcthError


def tkcth_message_from_bytes(data: bytes) -> TkcthMessage:
    """Decode bytes into a concrete TKCTH message by dispatching on ``type``."""
    obj = TkcthMessageBase._decode_bytes_to_obj(data)
    if not isinstance(obj, dict):
        raise ValueError("TKCTH message JSON must be an object")
    TkcthMessageBase._validate_message_dict(obj)
    typ = obj["type"]
    if typ == "token_delta":
        return TkcthTokenDelta.from_dict(obj)
    if typ == "finish":
        return TkcthFinish.from_dict(obj)
    if typ == "error":
        return TkcthError.from_dict(obj)
    raise ValueError(f"unknown TKCTH message type: {typ!r}")
