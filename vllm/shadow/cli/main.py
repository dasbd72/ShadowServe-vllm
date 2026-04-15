# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow process for accelerated serverless cold-start in vLLM.

This script is executed when the `--enable-shadow-migration-cold` flag is set.
It is designed to run concurrently with the main GPU-driven `vllm-serve`, handling
CPU-based decoding to bridge the latency gap while the main process initializes
(model loading, compilation, CUDA graph capture, and warmup).

Lifecycle:

- Parse CLI and construct ``ShadowEngine`` (loads CPU weights; can take a long time).
- Start the engine thread: bind a unique KVHTC UDS at
  ``{--shadow-kvhtc-ipc-prefix}-{16_hex}``, then write ``--ready-file`` with two lines:
  process PID, then the full socket path (see ``ShadowEngine._emit_ready``).
- Block in ``join()`` until migration+decode finishes or SIGTERM/SIGINT.
"""

import argparse
import logging
import os
import signal
import tempfile

from vllm.shadow.runtime.config import ShadowConfig
from vllm.shadow.runtime.engine import ShadowEngine

logger = logging.getLogger("vllm.shadow.cli.main")

# ── CLI ────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM shadow CPU process (serverless cold-start)",
    )
    parser.add_argument(
        "--shadow-kvhtc-ipc-prefix",
        dest="shadow_kvhtc_ipc_prefix",
        required=True,
        help="Path prefix for the shadow KVHTC Unix socket; `-` plus 16 hex chars "
        "are appended at startup (one unique socket per shadow process). "
        "Example: /tmp/vllm-shadow-kvhtc → /tmp/vllm-shadow-kvhtc-a1b2c3d4e5f67890. "
        "Required.",
    )
    parser.add_argument(
        "--ready-file",
        default=os.path.join(tempfile.gettempdir(), "vllm_shadow_ready"),
        help=(
            "Path written after KVHTC listen is ready: line 1 = PID, line 2 = full "
            "KVHTC socket path (same format as tests and docker smoke expect)."
        ),
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
        ready_file=args.ready_file,
        shadow_kvhtc_ipc_prefix=args.shadow_kvhtc_ipc_prefix,
        shadow_config=shadow_cfg,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating shutdown.", sig_name)
        shadow_engine.stop(reason=f"signal={sig_name}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    shadow_engine.start()
    shadow_engine.join()


if __name__ == "__main__":
    main()
