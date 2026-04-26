# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations


class Config:
    # Cap batch size when posting to /shadow_migration/migrate (avoid OOM).
    MAX_MIGRATION_IDS = 16

    HTTP_SHORT_TIMEOUT_S = 2.0
    HTTP_LONG_TIMEOUT_S = 60.0
    OPENAI_STREAM_ID_TIMEOUT_S = 30.0
    OPENAI_STREAM_ID_INTERVAL_S = 0.05

    SSE_DATA_PREFIX = "data: "

    DEFAULT_SMOKE_PROMPT = (
        "You are a meticulous technical writer. Write a detailed, multi-section "
        "explanation of how a serverless LLM system can migrate an in-flight "
        "request from a hot GPU process to a cold shadow CPU process, and later "
        "hand off to a cold GPU process.\n\n"
        "Requirements:\n"
        "- Use at least 10 sections with headings.\n"
        "- Include concrete step-by-step flows, failure modes, and recovery.\n"
        "- Include a glossary.\n"
        "- Include a short pseudo-protocol for the handoff envelope.\n"
        "- Keep generating content; do not stop early.\n\n"
        "Begin with a short overview, then go deep."
    )
