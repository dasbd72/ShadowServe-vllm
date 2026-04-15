#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Container entrypoint for the vllm-openai image.
#
# Without --enable-shadow-migration-cold this behaves identically to the previous
#   ENTRYPOINT ["vllm", "serve"]
#
# With --enable-shadow-migration-cold the entrypoint additionally launches a shadow
# CPU process (`python -m vllm.shadow.cli.main`) in the background before
# starting the main GPU server.  The shadow process is intended for
# serverless cold-start acceleration (Phase 1 skeleton).
#
# Resource-isolation knobs (best-effort, optional):
#   VLLM_SHADOW_CPUSET  – CPU affinity mask for the shadow process
#                          (e.g. "0-3").  Requires `taskset`.
#   VLLM_MAIN_CPUSET    – CPU affinity mask for the main vllm serve
#                          process.  Requires `taskset`.
#
# IPC knobs forwarded to the shadow process:
#   VLLM_SHADOW_KVHTC_IPC_PREFIX – UDS path prefix; shadow appends `-` + random hex.

set -euo pipefail

SHADOW_PID=""
ENABLE_SHADOW_MIGRATION_COLD=0
SHADOW_KVHTC_IPC_PREFIX=""

# ── Parse and strip cold-only flags from argv ──────────────────────
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --enable-shadow-migration-cold)
            ENABLE_SHADOW_MIGRATION_COLD=1
            ;;
        --shadow-kvhtc-ipc-prefix)
            # handled below (expects value in next token)
            ARGS+=("$arg")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Re-scan args to capture --shadow-kvhtc-ipc-prefix value and remove the pair from ARGS.
MAIN_ARGS=()
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    if [[ "${ARGS[$i]}" == "--shadow-kvhtc-ipc-prefix" ]]; then
        if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
            echo "[entrypoint] ERROR: --shadow-kvhtc-ipc-prefix requires a value" >&2
            exit 2
        fi
        SHADOW_KVHTC_IPC_PREFIX="${ARGS[$((i + 1))]}"
        i=$((i + 2))
        continue
    fi
    MAIN_ARGS+=("${ARGS[$i]}")
    i=$((i + 1))
done

# ── Signal handling ────────────────────────────────────────────────
cleanup() {
    if [[ -n "$SHADOW_PID" ]] && kill -0 "$SHADOW_PID" 2>/dev/null; then
        echo "[entrypoint] Forwarding signal to shadow CPU process (PID $SHADOW_PID)..."
        kill -TERM "$SHADOW_PID" 2>/dev/null || true
        wait "$SHADOW_PID" 2>/dev/null || true
        echo "[entrypoint] Shadow CPU process exited."
    fi
}
trap cleanup EXIT SIGTERM SIGINT

# ── Optionally start shadow CPU process ────────────────────────────
if [[ "$ENABLE_SHADOW_MIGRATION_COLD" -eq 1 ]]; then
    SHADOW_CMD=(python3 -m vllm.shadow.cli.main)

    # Forward IPC config to shadow process via CLI args
    if [[ -n "${SHADOW_KVHTC_IPC_PREFIX:-}" ]]; then
        SHADOW_CMD+=(--shadow-kvhtc-ipc-prefix "$SHADOW_KVHTC_IPC_PREFIX")
    elif [[ -n "${VLLM_SHADOW_KVHTC_IPC_PREFIX:-}" ]]; then
        SHADOW_CMD+=(--shadow-kvhtc-ipc-prefix "$VLLM_SHADOW_KVHTC_IPC_PREFIX")
    fi

    # Mirror model/math flags from ``vllm serve`` so the shadow CPU process loads the same HF config.
    i=0
    while [[ $i -lt ${#MAIN_ARGS[@]} ]]; do
        case "${MAIN_ARGS[$i]}" in
            --model|--dtype|--max-model-len|--block-size)
                if [[ $((i + 1)) -lt ${#MAIN_ARGS[@]} ]]; then
                    SHADOW_CMD+=("${MAIN_ARGS[$i]}" "${MAIN_ARGS[$((i + 1))]}")
                    i=$((i + 2))
                    continue
                fi
                ;;
        esac
        i=$((i + 1))
    done

    # Best-effort CPU affinity for shadow process
    if [[ -n "${VLLM_SHADOW_CPUSET:-}" ]] && command -v taskset &>/dev/null; then
        SHADOW_CMD=(taskset -c "$VLLM_SHADOW_CPUSET" "${SHADOW_CMD[@]}")
    fi

    echo "[entrypoint] Starting shadow CPU process: ${SHADOW_CMD[*]}"
    VLLM_TARGET_DEVICE=cpu "${SHADOW_CMD[@]}" &
    SHADOW_PID=$!
    echo "[entrypoint] Shadow CPU process started (PID $SHADOW_PID)."
fi

# ── Launch main GPU server ─────────────────────────────────────────
MAIN_CMD=(vllm serve "${MAIN_ARGS[@]}")

if [[ -n "${VLLM_MAIN_CPUSET:-}" ]] && command -v taskset &>/dev/null; then
    MAIN_CMD=(taskset -c "$VLLM_MAIN_CPUSET" "${MAIN_CMD[@]}")
fi

echo "[entrypoint] Starting main GPU server: ${MAIN_CMD[*]}"
exec "${MAIN_CMD[@]}"
