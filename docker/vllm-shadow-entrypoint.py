#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Python version of docker/vllm-shadow-entrypoint.sh
#
# - Without --shadow-cpu-enabled: behaves like `vllm serve ...`
# - With --shadow-cpu-enabled: also starts a shadow CPU process
#   (`python -m vllm.shadow.cli.main`) before launching the main server.
#
# Resource isolation (best-effort, optional):
#   VLLM_SHADOW_CPUSET / VLLM_MAIN_CPUSET (requires `taskset`)
#
# Optional PID files:
#   VLLM_SHADOW_PID_FILE / VLLM_MAIN_PID_FILE
#
# Shadow stdlib HTTP (``GET /health``, ``GET /shadow_migration/recv``) overrides:
#   VLLM_SHADOW_HTTP_HOST, VLLM_SHADOW_HTTP_PORT
#   (appended to ``python -m vllm.shadow.cli.main`` when set; else shadow CLI
#   defaults 127.0.0.1:8004).
#   Shadow-only: ``--shadow-http-host`` / ``--shadow-http-port`` may appear next to
#   ``--shadow-cpu-enabled``; they are stripped from ``vllm serve`` and forwarded
#   to the shadow process only.
#
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress


def _write_pid_file(path: str, pid: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{pid}\n")


def _rm_pid_file(path: str | None) -> None:
    if not path:
        return
    with suppress(FileNotFoundError):
        os.remove(path)


def _split_args(
    argv: Sequence[str],
) -> tuple[bool, str, str | None, str | None, list[str]]:
    """Return (shadow_cpu_enabled, shadow_kvhts_ipc_prefix, shadow_http_host,
    shadow_http_port, main_args). Shadow-only flags are removed from *main_args*."""
    shadow_cpu_enabled = False

    # First pass: strip shadow-only flag.
    args: list[str] = []
    for arg in argv:
        if arg == "--shadow-cpu-enabled":
            shadow_cpu_enabled = True
        else:
            args.append(arg)

    # Second pass: extract and remove shadow-only flags (not valid for
    # ``vllm serve``); forward only to the shadow process.
    shadow_kvhts_ipc_prefix = ""
    shadow_http_host: str | None = None
    shadow_http_port: str | None = None
    main_args: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--shadow-kvhts-ipc-prefix":
            if i + 1 >= len(args):
                print(
                    "[entrypoint] ERROR: --shadow-kvhts-ipc-prefix requires a value",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(2)
            shadow_kvhts_ipc_prefix = args[i + 1]
            i += 2
            continue
        if args[i] == "--shadow-http-host":
            if i + 1 >= len(args):
                print(
                    "[entrypoint] ERROR: --shadow-http-host requires a value",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(2)
            shadow_http_host = args[i + 1]
            i += 2
            continue
        if args[i] == "--shadow-http-port":
            if i + 1 >= len(args):
                print(
                    "[entrypoint] ERROR: --shadow-http-port requires a value",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(2)
            shadow_http_port = args[i + 1]
            i += 2
            continue
        main_args.append(args[i])
        i += 1

    return (
        shadow_cpu_enabled,
        shadow_kvhts_ipc_prefix,
        shadow_http_host,
        shadow_http_port,
        main_args,
    )


def _shadow_common_flags_from_main(main_args: Sequence[str]) -> list[str]:
    """Mirror model/math flags from `vllm serve` into the shadow CLI."""
    mirrored = []
    i = 0
    while i < len(main_args):
        if main_args[i] in (
            "--log-level",
            "--model",
            "--dtype",
            "--max-model-len",
            "--block-size",
        ) and i + 1 < len(main_args):
            mirrored.extend([main_args[i], main_args[i + 1]])
            i += 2
            continue
        i += 1
    return mirrored


def _append_shadow_http_flags(
    cmd: list[str],
    host_from_argv: str | None,
    port_from_argv: str | None,
) -> None:
    """Apply argv wins over env; unset uses shadow CLI built-in defaults."""
    h = (host_from_argv or os.environ.get("VLLM_SHADOW_HTTP_HOST", "") or "").strip()
    p = (port_from_argv or os.environ.get("VLLM_SHADOW_HTTP_PORT", "") or "").strip()
    if h:
        cmd.extend(["--shadow-http-host", h])
    if p:
        cmd.extend(["--shadow-http-port", p])


def _maybe_taskset_prefix(cmd: list[str], cpuset_env: str) -> list[str]:
    cpuset = os.environ.get(cpuset_env, "").strip()
    if not cpuset:
        return cmd
    if shutil.which("taskset") is None:
        return cmd
    return ["taskset", "-c", cpuset, *cmd]


def _popen(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen:
    # Start a new process group so we can forward signals robustly.
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )


def _terminate_proc(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        print(
            f"[entrypoint] Forwarding signal to {name} (PID {proc.pid})...",
            flush=True,
        )
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _wait_proc(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return
    try:
        proc.wait()
        print(f"[entrypoint] {name} exited.", flush=True)
    except Exception:
        # Best-effort; entrypoints should not crash during cleanup.
        pass


def main() -> int:
    shadow_proc: subprocess.Popen | None = None
    main_proc: subprocess.Popen | None = None

    shadow_pid_file = os.environ.get("VLLM_SHADOW_PID_FILE")
    main_pid_file = os.environ.get("VLLM_MAIN_PID_FILE")

    (
        shadow_cpu_enabled,
        shadow_kvhts_ipc_prefix,
        shadow_http_host,
        shadow_http_port,
        main_args,
    ) = _split_args(sys.argv[1:])

    def cleanup(_signum: int | None = None, _frame=None) -> None:
        nonlocal shadow_proc, main_proc
        if main_proc is not None and main_proc.poll() is None:
            _terminate_proc(main_proc, "main GPU server")
            _wait_proc(main_proc, "Main GPU server")
        if shadow_proc is not None and shadow_proc.poll() is None:
            _terminate_proc(shadow_proc, "shadow CPU process")
            _wait_proc(shadow_proc, "Shadow CPU process")
        _rm_pid_file(main_pid_file)
        _rm_pid_file(shadow_pid_file)

    # Handle SIGTERM/SIGINT like the shell trap.
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    if shadow_cpu_enabled:
        shadow_cmd = ["python3", "-m", "vllm.shadow.cli.main"]
        if shadow_kvhts_ipc_prefix:
            shadow_cmd += ["--shadow-kvhts-ipc-prefix", shadow_kvhts_ipc_prefix]
        shadow_cmd += _shadow_common_flags_from_main(main_args)
        _append_shadow_http_flags(shadow_cmd, shadow_http_host, shadow_http_port)
        shadow_cmd = _maybe_taskset_prefix(shadow_cmd, "VLLM_SHADOW_CPUSET")

        shadow_env = dict(os.environ)
        shadow_env["VLLM_TARGET_DEVICE"] = "cpu"

        print(
            f"[entrypoint] Starting shadow CPU process: {' '.join(shadow_cmd)}",
            flush=True,
        )
        shadow_proc = _popen(shadow_cmd, env=shadow_env)
        print(
            f"[entrypoint] Shadow CPU process started (PID {shadow_proc.pid}).",
            flush=True,
        )
        if shadow_pid_file:
            _write_pid_file(shadow_pid_file, shadow_proc.pid)

    main_cmd = ["vllm", "serve", *main_args]
    main_cmd = _maybe_taskset_prefix(main_cmd, "VLLM_MAIN_CPUSET")

    print(f"[entrypoint] Starting main GPU server: {' '.join(main_cmd)}", flush=True)
    main_proc = _popen(main_cmd)
    if main_pid_file:
        _write_pid_file(main_pid_file, main_proc.pid)

    # If the main process exits (crash or clean shutdown), also stop shadow.
    try:
        rc = main_proc.wait()
    finally:
        main_proc = None
        _rm_pid_file(main_pid_file)

        if shadow_proc is not None and shadow_proc.poll() is None:
            print(
                "[entrypoint] Main GPU server exited; stopping shadow CPU process "
                f"(PID {shadow_proc.pid})...",
                flush=True,
            )
            _terminate_proc(shadow_proc, "shadow CPU process")
            _wait_proc(shadow_proc, "Shadow CPU process")
        shadow_proc = None
        _rm_pid_file(shadow_pid_file)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
