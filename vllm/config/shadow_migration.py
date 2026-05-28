# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for serverless hot GPU → shadow CPU KV migration."""

from vllm.config.utils import config


@config
class ShadowMigrationConfig:
    """Settings for shadow KV migration (separate from upstream KVConnector / P&D)."""

    shadow_sender_enabled: bool = False
    """When True, engine and API may run shadow migration (default off)."""

    shadow_receiver_enabled: bool = False
    """When True, engine and API may receive shadow migration (default off)."""
