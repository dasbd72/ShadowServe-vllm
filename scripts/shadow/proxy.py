# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from client import StreamingCompletionClient
from config import Config

logger = logging.getLogger("scripts.shadow.proxy")


def _wait_kvstc_migration_completed(
    base_url: str,
    migration_id: int,
    *,
    timeout_s: float,
    interval_s: float,
) -> None:
    url = f"{base_url}/shadow_migration/completed"
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        try:
            with contextlib.closing(
                requests.get(url, timeout=Config.HTTP_SHORT_TIMEOUT_S)
            ) as r:
                r.raise_for_status()
                data = r.json()
                completed_ids = data.get("completed_kvstc_sessions", [])
                if migration_id in completed_ids:
                    return
                raise RuntimeError("Not completed yet")
        except (OSError, RuntimeError, ValueError, requests.RequestException):
            time.sleep(interval_s)
    raise RuntimeError(f"migration completed check failed for {base_url}")


class StreamingCompletionProxy:
    """Streaming ``/v1/completions`` proxy: one worker thread owns the HTTP stream."""

    def __init__(
        self,
        base_url: str,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        http_timeout_s: float = Config.HTTP_LONG_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._prompt = prompt
        self._remaining_max_tokens = max_tokens
        self._temperature = temperature
        self._http_timeout_s = http_timeout_s

        self._migrated_clients: list[StreamingCompletionClient] = []
        self._client = StreamingCompletionClient(
            base_url,
            model,
            prompt,
            max_tokens=self._remaining_max_tokens,
            temperature=temperature,
            http_timeout_s=http_timeout_s,
        )

        # Only concat text after migration
        self._output_text: str = ""
        self._pivots: list[int] = []

    def start(self) -> None:
        self._client.start()

    def cancel(self) -> None:
        self._client.cancel()

    def join(self, timeout: float | None = None) -> None:
        """Wait until the client and all migrated clients finish."""
        with ThreadPoolExecutor(
            max_workers=len(self._migrated_clients) + 1
        ) as executor:
            for client in self._migrated_clients:
                executor.submit(client.join, timeout=timeout)
            executor.submit(self._client.join, timeout=timeout)

    def wait_openai_id(self) -> str:
        return self._client.wait_openai_id()

    def record_pivot(self) -> None:
        self._pivots.append(len(self._output_text) + len(self._client.output_text))

    def migrate(self, base_url: str, migration_id: int) -> None:
        """Migrate the client to a new base URL."""
        _wait_kvstc_migration_completed(
            base_url,
            migration_id,
            timeout_s=200,
            interval_s=0.1,
        )

        client = self._client

        if client.finish_reason is not None and client.finish_reason != "migrated":
            logger.warning(
                "Client already finished, finish reason: %s, skipping migration",
                client.finish_reason,
            )
            return

        client.cancel()
        client_output_text = client.output_text
        used_tokens = client.completion_tokens
        self._pivots.append(len(self._output_text) + len(client_output_text))
        self._remaining_max_tokens = max(0, self._remaining_max_tokens - used_tokens)
        self._prompt += client_output_text
        self._output_text += client_output_text

        self._migrated_clients.append(client)
        self._client = StreamingCompletionClient(
            base_url,
            self._model,
            self._prompt,
            max_tokens=self._remaining_max_tokens,
            temperature=self._temperature,
            http_timeout_s=self._http_timeout_s,
        )
        self._client.start()

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def output_text(self) -> str:
        return self._output_text + self._client.output_text

    @property
    def pivots(self) -> list[int]:
        return self._pivots.copy()
