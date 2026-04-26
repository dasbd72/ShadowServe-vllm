#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from config import Config
from proxy import StreamingCompletionProxy
from server import VllmServer

logger = logging.getLogger("scripts.shadow.main")


@dataclass(frozen=True)
class SmokePaths:
    """Log paths and IPC prefixes under a single working directory."""

    root: Path
    kvhts_ipc_prefix: str
    tksth_ipc_prefix: str
    kvstc_ipc_prefix: str
    hot_log: Path
    cold_log: Path
    shadow_log: Path

    @classmethod
    def under(cls, root: Path) -> SmokePaths:
        return cls(
            root=root,
            kvhts_ipc_prefix=str(root / "vllm-shadow-kvhts"),
            tksth_ipc_prefix=str(root / "vllm-hot-tksth"),
            kvstc_ipc_prefix=str(root / "vllm-cold-kvstc"),
            hot_log=root / "hot.log",
            cold_log=root / "cold.log",
            shadow_log=root / "shadow.log",
        )


def _get_shadow_kvhts_ipc_path(base_url: str) -> str:
    url = f"{base_url}/shadow_migration/recv"
    with contextlib.closing(
        requests.get(url, timeout=Config.HTTP_SHORT_TIMEOUT_S)
    ) as r:
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"HTTP GET {url} returned invalid JSON: {data}")
    kvhts_ipc_path = data.get("kvhts_ipc_path")
    if not isinstance(kvhts_ipc_path, str) or not kvhts_ipc_path:
        raise RuntimeError(
            f"HTTP GET {url} returned invalid kvhts_ipc_path: {kvhts_ipc_path}"
        )
    return kvhts_ipc_path


def _get_running_internal_ids(base_url: str, openai_ids: list[str]) -> list[str]:
    url = f"{base_url}/shadow_migration/requests"
    with contextlib.closing(
        requests.get(url, timeout=Config.HTTP_SHORT_TIMEOUT_S)
    ) as r:
        r.raise_for_status()
        data = r.json()
    out: list[str] = []
    for info in data.get("active", []):
        if not isinstance(info, dict):
            continue
        iid = str(info.get("request_id", ""))
        if str(info.get("status", "")) != "RUNNING":
            continue
        if any(oid and oid in iid for oid in openai_ids):
            out.append(iid)
    return out


def _post_migrate(
    base_url: str, kvhts_ipc_path: str, request_ids: list[str]
) -> tuple[int, list[str]]:
    url = f"{base_url}/shadow_migration/migrate"
    with contextlib.closing(
        requests.post(
            url,
            json={
                "request_ids": request_ids,
                "kvhts_ipc_path": kvhts_ipc_path,
            },
            headers={"Content-Type": "application/json"},
            timeout=Config.HTTP_SHORT_TIMEOUT_S,
        )
    ) as r:
        r.raise_for_status()
        data = r.json()
    migration_id = data.get("migration_id", None)
    if not isinstance(migration_id, int):
        raise RuntimeError(
            f"HTTP POST {url} returned invalid migration_id: {migration_id}"
        )
    migrated_request_ids = data.get("migrated_request_ids", [])
    if not isinstance(migrated_request_ids, list):
        raise RuntimeError(
            f"HTTP POST {url} returned invalid "
            f"migrated_request_ids: {migrated_request_ids}"
        )
    return migration_id, migrated_request_ids


# --- Child processes ---


def _hot_cold_and_shadow_commands(
    model: str,
    dtype: str,
    hot_host: str,
    hot_port: int,
    additional_blocks_per_request: int,
    cold_host: str,
    cold_port: int,
    shadow_http_host: str,
    shadow_http_port: int,
    paths: SmokePaths,
    cold_base_url: str,
) -> tuple[list[str], list[str], list[str]]:
    py = sys.executable
    hot_cmd = [
        py,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        hot_host,
        "--port",
        str(hot_port),
        "--dtype",
        dtype,
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.80",
        "--shadow-sender-enabled",
        "--shadow-additional-blocks-per-request",
        str(additional_blocks_per_request),
        "--shadow-tksth-ipc-prefix",
        paths.tksth_ipc_prefix,
    ]
    cold_cmd = [
        py,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        cold_host,
        "--port",
        str(cold_port),
        "--dtype",
        dtype,
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.80",
        "--shadow-receiver-enabled",
        "--shadow-kvstc-ipc-prefix",
        paths.kvstc_ipc_prefix,
    ]
    shadow_cmd = [
        py,
        "-m",
        "vllm.shadow.cli.main",
        "--shadow-kvhts-ipc-prefix",
        paths.kvhts_ipc_prefix,
        "--cold-base-api-url",
        cold_base_url,
        "--model",
        model,
        "--dtype",
        dtype,
        "--block-size",
        "16",
        "--log-level",
        "INFO",
        "--shadow-http-host",
        shadow_http_host,
        "--shadow-http-port",
        str(shadow_http_port),
    ]
    return hot_cmd, cold_cmd, shadow_cmd


def _hot_cold_and_shadow_env(
    hot_cuda_visible_devices: str | None,
    cold_cuda_visible_devices: str | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    hot_env = os.environ.copy()
    if hot_cuda_visible_devices is not None:
        hot_env["CUDA_VISIBLE_DEVICES"] = str(hot_cuda_visible_devices)

    cold_env = os.environ.copy()
    if cold_cuda_visible_devices is not None:
        cold_env["CUDA_VISIBLE_DEVICES"] = str(cold_cuda_visible_devices)

    shadow_env = os.environ.copy()
    shadow_env["CUDA_VISIBLE_DEVICES"] = ""
    shadow_env["VLLM_TARGET_DEVICE"] = "cpu"
    return hot_env, cold_env, shadow_env


class Executor:
    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        hot_host: str,
        hot_port: int,
        cold_host: str,
        cold_port: int,
        shadow_http_host: str,
        shadow_http_port: int,
        hot_cuda_visible_devices: str | None,
        cold_cuda_visible_devices: str | None,
        additional_blocks_per_request: int,
        num_requests: int,
        prompt: str,
        completion_max_tokens: int,
        paths: SmokePaths,
    ):
        self.model = model
        self.dtype = dtype

        self.hot_host = hot_host
        self.hot_port = hot_port
        self.cold_host = cold_host
        self.cold_port = cold_port
        self.shadow_http_host = shadow_http_host
        self.shadow_http_port = shadow_http_port
        self.hot_base_url = f"http://{hot_host}:{hot_port}"
        self.shadow_base_url = f"http://{shadow_http_host}:{shadow_http_port}"
        self.cold_base_url = f"http://{cold_host}:{cold_port}"

        self.hot_cuda_visible_devices = hot_cuda_visible_devices
        self.cold_cuda_visible_devices = cold_cuda_visible_devices
        self.additional_blocks_per_request = additional_blocks_per_request
        self.num_requests = num_requests
        self.prompt = prompt
        self.completion_max_tokens = completion_max_tokens
        self.paths = paths

        self.stop_event = threading.Event()

        self.hot_cmd, self.cold_cmd, self.shadow_cmd = _hot_cold_and_shadow_commands(
            self.model,
            self.dtype,
            self.hot_host,
            self.hot_port,
            self.additional_blocks_per_request,
            self.cold_host,
            self.cold_port,
            self.shadow_http_host,
            self.shadow_http_port,
            self.paths,
            self.cold_base_url,
        )
        self.hot_env, self.cold_env, self.shadow_env = _hot_cold_and_shadow_env(
            self.hot_cuda_visible_devices,
            self.cold_cuda_visible_devices,
        )
        self.hot_server: VllmServer | None = None
        self.cold_server: VllmServer | None = None
        self.shadow_server: VllmServer | None = None

        self.kvhts_ipc_path: str | None = None
        self.clients: list[StreamingCompletionProxy] = []
        self.openai_ids: list[str] = []
        self.migration_id: int | None = None
        self.running_internal_ids: list[str] = []
        self.migrated_internal_ids: list[str] = []

    def run(self) -> None:
        self._start_hot_process()
        self._start_streaming_clients()

        # Start shadow migration
        self._start_cold_and_shadow_processes()
        self.kvhts_ipc_path = _get_shadow_kvhts_ipc_path(self.shadow_base_url)
        running_internal_ids = _get_running_internal_ids(
            self.hot_base_url,
            self.openai_ids,
        )
        self.running_internal_ids = running_internal_ids[: Config.MAX_MIGRATION_IDS]

        logger.info("Posting migration...")
        self._record_pivots()
        self.migration_id, self.migrated_internal_ids = _post_migrate(
            self.hot_base_url,
            self.kvhts_ipc_path,
            self.running_internal_ids,
        )
        if not self.migrated_internal_ids:
            raise RuntimeError(
                "migration returned no migrated_request_ids; "
                f"expected at least one of {self.running_internal_ids!r}. See hot log."
            )
        if set(self.migrated_internal_ids) != set(self.running_internal_ids):
            raise RuntimeError(
                "migration response mismatch: expected ids "
                f"{set(self.running_internal_ids)!r}, "
                f"got {set(self.migrated_internal_ids)!r}"
            )

        self.cold_server.wait_health(timeout_s=200, interval_s=0.1)
        logger.info("Cold server initialized.")

        self._migrate_clients()
        logger.info("Clients migrated.")

        logger.info("Waiting 5s after clients migrated...")
        time.sleep(5)

        self._print_outputs()
        logger.info("Executor run completed.")

    def terminate(self) -> None:
        self._cancel_and_join_clients()
        self._terminate_processes()

    def _start_hot_process(self) -> None:
        self.hot_server = VllmServer(
            self.hot_cmd, self.hot_env, self.paths.hot_log, self.hot_base_url
        )
        self.hot_server.start()
        logger.info("Starting hot process. hot log: %s", self.paths.hot_log)
        self.hot_server.wait_health(timeout_s=200, interval_s=0.1)
        logger.info("Hot server initialized.")

    def _start_streaming_clients(self) -> None:
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_requests) as executor:
            for i in range(self.num_requests):
                futures.append(executor.submit(self._start_streaming_client, i))
            for future in futures:
                c, openai_id = future.result()
                self.clients.append(c)
                self.openai_ids.append(openai_id)

    def _start_streaming_client(self, i: int) -> None:
        c = StreamingCompletionProxy(
            self.hot_base_url,
            self.model,
            f"This is request {i}. {self.prompt}",
            max_tokens=self.completion_max_tokens,
            temperature=0.0,
        )
        c.start()
        return c, c.wait_openai_id()

    def _start_cold_and_shadow_processes(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self._start_cold_process)
            executor.submit(self._start_shadow_process)

    def _start_cold_process(self) -> None:
        self.cold_server = VllmServer(
            self.cold_cmd,
            self.cold_env,
            self.paths.cold_log,
            self.cold_base_url,
        )
        logger.info("Starting cold process. cold log: %s", self.paths.cold_log)
        self.cold_server.start()

    def _start_shadow_process(self) -> None:
        self.shadow_server = VllmServer(
            self.shadow_cmd,
            self.shadow_env,
            self.paths.shadow_log,
            self.shadow_base_url,
        )
        self.shadow_server.start()
        logger.info("Starting shadow process. shadow log: %s", self.paths.shadow_log)
        self.shadow_server.wait_health(timeout_s=100, interval_s=0.1)
        logger.info("Shadow server initialized.")

    def _record_pivots(self) -> None:
        for c in self.clients:
            c.record_pivot()

    def _migrate_clients(self) -> None:
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            for c in self.clients:
                executor.submit(c.migrate, self.cold_base_url, self.migration_id)

    def _print_outputs(self) -> None:
        for oid, c in zip(self.openai_ids, self.clients, strict=True):
            logger.info("--------------------------------")
            text = c.output_text
            pivots = c.pivots
            if pivots and pivots[0] != 0:
                pivots = [0] + pivots
            if pivots and pivots[-1] != len(text):
                pivots = pivots + [len(text)]
            text_chunks = [
                text[pivots[i] : pivots[i + 1]] for i in range(len(pivots) - 1)
            ]
            text = " | ".join(text_chunks)
            logger.info("Request %s", oid)
            logger.info("Prompt: %s", c.prompt)
            logger.info("Output Text: %s", text)
            logger.info("Pivots: %s", pivots)

    def _cancel_and_join_clients(self) -> None:
        if not self.clients:
            return
        clients = self.clients
        self.clients = []
        for c in clients:
            c.cancel()
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            for c in clients:
                executor.submit(c.join, timeout=2.0)

    def _terminate_processes(self) -> None:
        servers = [self.hot_server, self.shadow_server, self.cold_server]
        self.hot_server = None
        self.shadow_server = None
        self.cold_server = None
        servers = [server for server in servers if server is not None]
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            for server in servers:
                executor.submit(server.terminate)
        logger.info("Processes terminated.")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python test_shadow_migration_smoke.py "
            "--model meta-llama/Llama-3.2-1B-Instruct --hot-port 8123\n"
            "\n"
            "The shadow subprocess is spawned with the same --model, --dtype, and "
            "--block-size as vllm serve (required by vllm.shadow.cli.main)."
        ),
    )
    ap.add_argument(
        "--hot-cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES override for the hot vLLM process.",
    )
    ap.add_argument(
        "--cold-cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES override for the cold vLLM process.",
    )
    ap.add_argument(
        "--additional-blocks-per-request",
        type=int,
        default=10,
        help="Additional blocks per request for the shadow vLLM process.",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VLLM_TEST_MODEL", "facebook/opt-125m"),
    )
    ap.add_argument("--dtype", default="bfloat16")

    ap.add_argument("--hot-host", default="127.0.0.1")
    ap.add_argument("--hot-port", type=int, default=8123)
    ap.add_argument(
        "--cold-host",
        default="127.0.0.1",
        help="Cold vLLM host.",
    )
    ap.add_argument(
        "--cold-port",
        type=int,
        default=8124,
        help="Cold vLLM port (separate from hot).",
    )
    ap.add_argument(
        "--shadow-http-host",
        default="127.0.0.1",
        help="Must match shadow subprocess --shadow-http-host (GET /health, /recv).",
    )
    ap.add_argument(
        "--shadow-http-port",
        type=int,
        default=8004,
        help="Must match shadow subprocess --shadow-http-port.",
    )

    ap.add_argument("--num-requests", type=int, default=1)
    ap.add_argument("--prompt", default=Config.DEFAULT_SMOKE_PROMPT)
    ap.add_argument("--completion-max-tokens", type=int, default=1024)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    workdir = Path(tempfile.mkdtemp(dir="/tmp/vllm-shadow-smoke-run"))
    logger.info("workdir: %s", workdir)
    paths = SmokePaths.under(workdir)

    executor = Executor(
        model=args.model,
        dtype=args.dtype,
        hot_host=args.hot_host,
        hot_port=args.hot_port,
        cold_host=args.cold_host,
        cold_port=args.cold_port,
        shadow_http_host=args.shadow_http_host,
        shadow_http_port=args.shadow_http_port,
        hot_cuda_visible_devices=args.hot_cuda_visible_devices,
        cold_cuda_visible_devices=args.cold_cuda_visible_devices,
        additional_blocks_per_request=args.additional_blocks_per_request,
        num_requests=args.num_requests,
        prompt=args.prompt,
        completion_max_tokens=args.completion_max_tokens,
        paths=paths,
    )

    try:
        with contextlib.suppress(KeyboardInterrupt):
            executor.run()
    finally:
        logger.info("Terminating processes from main...")
        executor.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
