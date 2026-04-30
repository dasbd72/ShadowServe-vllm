# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for serverless hot GPU → shadow CPU KV migration."""

from vllm.config.utils import config


@config
class ShadowMigrationConfig:
    """Settings for shadow KV migration (separate from upstream KVConnector / P&D)."""

    shadow_sender_enabled: bool = False
    """When True, engine and API may run shadow migration (default off)."""

    shadow_additional_blocks_per_request: int = 0
    """Extra shadow KV block slots per request after migrated GPU blocks."""

    shadow_tksth_ipc_prefix: str | None = None
    """Unix socket path prefix for the hot-side TKSTH listener (token stream back from
    shadow). A unique path is formed by appending a random suffix per migration batch,
    analogous to ``--shadow-kvhts-ipc-prefix`` on the shadow process.
    """

    shadow_receiver_enabled: bool = False
    """When True, engine and API may receive shadow migration (default off)."""
