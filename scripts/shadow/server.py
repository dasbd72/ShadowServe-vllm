# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import requests
from config import Config


class VllmServer:
    def __init__(
        self, cmd: list[str], env: dict[str, str], log_path: Path, base_url: str
    ):
        self._cmd = cmd
        self._env = env
        self._log_path = log_path
        self._base_url = base_url

        self._process: subprocess.Popen | None = None

    def start(self):
        if self._process is not None:
            raise RuntimeError("Server already started")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w", encoding="utf-8") as f:
            self._process = subprocess.Popen(
                self._cmd,
                env=self._env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

    def terminate(self, *, sigint_grace_s: float = 3.0, sigterm_grace_s: float = 5.0):
        if self._process is None:
            raise RuntimeError("Server not started")

        if self._process.poll() is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=1.0)
            return
        self._signal_process_tree(signal.SIGINT)
        deadline_s = time.monotonic() + sigint_grace_s
        while time.monotonic() < deadline_s and self._process.poll() is None:
            time.sleep(0.05)
        if self._process.poll() is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=2.0)
            return
        self._signal_process_tree(signal.SIGTERM)
        try:
            self._process.wait(timeout=sigterm_grace_s)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass
        self._signal_process_tree(signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(timeout=10.0)

    def _signal_process_tree(self, sig: int):
        if self._process.pid is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(self._process.pid, sig)
            return

    def wait_health(self, *, timeout_s: float, interval_s: float) -> None:
        deadline_s = time.monotonic() + timeout_s
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline_s:
            try:
                with contextlib.closing(
                    requests.get(url, timeout=Config.HTTP_SHORT_TIMEOUT_S)
                ) as r:
                    if r.status_code == 200:
                        return
                    raise RuntimeError(f"HTTP GET {url} returned {r.status_code}")
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                time.sleep(interval_s)

        raise RuntimeError(f"health check timed out for {self._base_url}")
