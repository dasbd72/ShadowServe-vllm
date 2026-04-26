# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import threading
import time

import requests
from config import Config


class StreamingCompletionClient:
    """Streaming ``/v1/completions`` client: one worker thread owns the HTTP stream."""

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

        self._lock = threading.Lock()
        self._openai_id: str | None = None
        self._text_chunks: list[str] = []
        self._error: BaseException | None = None

        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._resp: requests.Response | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="streaming-completion",
            daemon=True,
        )

    def start(self) -> None:
        if self._thread.ident is not None:
            raise RuntimeError("StreamingCompletionClient.start() called twice")
        self._thread.start()

    def _append_text(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            self._text_chunks.append(chunk)

    def _worker(self) -> None:
        try:
            self._resp = requests.post(
                f"{self._base_url}/v1/completions",
                json={
                    "model": self._model,
                    "prompt": self._prompt,
                    "max_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "stream": True,
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                timeout=self._http_timeout_s,
                stream=True,
            )
            self._resp.raise_for_status()
            for line in self._resp.iter_lines(
                decode_unicode=True,
                delimiter="\n",
            ):
                if self._cancel.is_set():
                    break
                if line is None:
                    continue
                line_st = (line or "").strip()
                if not line_st.startswith(Config.SSE_DATA_PREFIX):
                    continue
                payload = line_st.removeprefix(Config.SSE_DATA_PREFIX).strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                rid = ev.get("id")
                if isinstance(rid, str) and rid:
                    with self._lock:
                        if self._openai_id is None:
                            self._openai_id = rid
                choices = ev.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                c0 = choices[0]
                if not isinstance(c0, dict):
                    continue
                text = c0.get("text")
                if isinstance(text, str) and text:
                    self._append_text(text)
        except BaseException as e:
            with self._lock:
                self._error = e
        finally:
            with contextlib.suppress(Exception):
                if self._resp is not None:
                    self._resp.close()
            self._finished.set()

    def wait_openai_id(self) -> str:
        """Block until the SSE exposes an ``id``, the stream ends, or timeout."""
        deadline = time.time() + Config.OPENAI_STREAM_ID_TIMEOUT_S
        while time.time() < deadline:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError(
                        "streaming completions failed before id was seen"
                    ) from self._error
                if self._openai_id:
                    return self._openai_id
            if self._finished.is_set():
                break
            time.sleep(Config.OPENAI_STREAM_ID_INTERVAL_S)
        with self._lock:
            if self._openai_id:
                return self._openai_id
            err = self._error
        if err is not None:
            raise RuntimeError(
                "streaming completions failed before id was seen"
            ) from err
        raise RuntimeError("timeout waiting for streaming OpenAI request id")

    def get_output_text(self) -> str:
        """Snapshot of decoded completion text accumulated so far (thread-safe)."""
        with self._lock:
            out = "".join(self._text_chunks)
        return out

    def cancel(self) -> None:
        """Stop the stream: worker exits ``iter_lines`` and closes the response."""
        self._cancel.set()
        r = self._resp
        if r is not None:
            with contextlib.suppress(Exception):
                r.close()

    def join(self, timeout: float | None = None) -> None:
        """Wait until the worker thread finishes (stream complete, error, or cancel)."""
        self._thread.join(timeout=timeout)

    @property
    def finished(self) -> bool:
        return self._finished.is_set()
