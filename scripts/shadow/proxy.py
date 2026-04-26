# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from concurrent.futures import ThreadPoolExecutor

from client import StreamingCompletionClient
from config import Config

logger = logging.getLogger("scripts.shadow.proxy")


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
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http_timeout_s = http_timeout_s

        self._migrated_clients: list[StreamingCompletionClient] = []
        self._client = StreamingCompletionClient(
            base_url,
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            http_timeout_s=http_timeout_s,
        )
        self._text_chunks: list[str] = []

    def start(self) -> None:
        self._client.start()

    def wait_openai_id(self) -> str:
        return self._client.wait_openai_id()

    def migrate(self, base_url: str) -> None:
        """Migrate the client to a new base URL."""
        if self._client.finished:
            logger.warning("Client already finished, skipping migration")
            return
        client = self._client
        client.cancel()
        client_output_text = client.get_output_text()
        self._prompt += client_output_text
        self._text_chunks.append(client_output_text)
        self._migrated_clients.append(client)

        self._client = StreamingCompletionClient(
            base_url,
            self._model,
            self._prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            http_timeout_s=self._http_timeout_s,
        )
        self._client.start()

    def get_output_text(self) -> str:
        """Snapshot of decoded completion text accumulated so far (thread-safe)."""
        self._text_chunks.append(self._client.get_output_text())
        return "".join(self._text_chunks)

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

    @property
    def finished(self) -> bool:
        return self._client.finished
