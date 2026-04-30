# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow process for accelerated serverless cold-start in vLLM.

This script is executed when the `--shadow-cpu-enabled` flag is set.
It is designed to run concurrently with the main GPU-driven `vllm-serve`, handling
CPU-based decoding to bridge the latency gap while the main process initializes
(model loading, compilation, CUDA graph capture, and warmup).

Lifecycle:

- Parse CLI and construct ``ShadowEngine`` (loads CPU weights; can take a long time).
- Start the engine thread: bind a unique KVHTS UDS at
  ``{--shadow-kvhts-ipc-prefix}-{16_hex}`` (path exposed via shadow HTTP).
- Always serve ``GET /health`` and ``GET /shadow_migration/recv`` on the shadow stdlib
  HTTP server (default ``--shadow-http-host`` / ``--shadow-http-port`` 127.0.0.1:8004).
- Block in ``join()`` until migration+decode finishes or SIGTERM/SIGINT.
"""

import argparse
import logging
import os
import signal

from vllm.shadow.runtime.config import ShadowConfig
from vllm.shadow.runtime.engine import ShadowEngine

logger = logging.getLogger("vllm.shadow.cli.main")

# ── CLI ────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM shadow CPU process (serverless cold-start)",
    )
    parser.add_argument(
        "--shadow-kvhts-ipc-prefix",
        dest="shadow_kvhts_ipc_prefix",
        required=True,
        help="Path prefix for the shadow KVHTS Unix socket; `-` plus 16 hex chars "
        "are appended at startup (one unique socket per shadow process). "
        "Example: /tmp/vllm-shadow-kvhts → /tmp/vllm-shadow-kvhts-a1b2c3d4e5f67890. "
        "Required.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Same model id/path as ``vllm serve`` (loads HF config + safetensors).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Weight/KV dtype alignment with the GPU server (e.g. bfloat16).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=-1,
        help="Mirrors ``--max-model-len`` on the GPU process. Use -1 for auto.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV block size; must match the hot pod ``--block-size``.",
    )
    parser.add_argument(
        "--shadow-http-host",
        default="127.0.0.1",
        help=(
            "Bind address for the shadow stdlib HTTP server (GET /health, "
            "GET /shadow_migration/recv)."
        ),
    )
    parser.add_argument(
        "--shadow-http-port",
        type=int,
        default=8004,
        metavar="PORT",
        help=(
            "TCP port for the threaded stdlib HTTP server. "
            "GET /shadow_migration/recv returns 503 until the KVHTS listener is bound, "
            "then 200 with JSON ``kvhts_ipc_path``. Default: 8004."
        ),
    )
    return parser.parse_args(argv)


def _shadow_config_from_args(args: argparse.Namespace) -> ShadowConfig:
    model = str(args.model)
    return ShadowConfig(
        model=model,
        dtype=str(args.dtype),
        max_model_len=int(args.max_model_len),
        block_size=int(args.block_size),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    logger.info("Shadow CPU engine starting (PID %d)...", os.getpid())
    shadow_cfg = _shadow_config_from_args(args)
    shadow_engine = ShadowEngine(
        shadow_kvhts_ipc_prefix=args.shadow_kvhts_ipc_prefix,
        shadow_config=shadow_cfg,
        shadow_http_host=str(args.shadow_http_host),
        shadow_http_port=int(args.shadow_http_port),
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating shutdown.", sig_name)
        shadow_engine.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    shadow_engine.run()


if __name__ == "__main__":
    main()
