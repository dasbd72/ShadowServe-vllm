# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import threading
import time

import requests
from config import Config


class StreamingChatCompletionClient:
    """Streaming ``/v1/chat/completions`` client owned by one worker thread."""

    def __init__(
        self,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        temperature: float = 0.0,
        http_timeout_s: float = Config.HTTP_LONG_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._messages = [message.copy() for message in messages]
        self._max_completion_tokens = max_completion_tokens
        self._temperature = temperature
        self._http_timeout_s = http_timeout_s

        self._lock = threading.Lock()
        self._openai_id: str | None = None
        self._text_chunks: list[str] = []
        self._completion_tokens: int = 0
        self._finish_reason: str | None = None
        self._error: Exception | None = None

        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._resp: requests.Response | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="streaming-chat-completion",
            daemon=True,
        )

    def start(self) -> None:
        if self._thread.ident is not None:
            raise RuntimeError("StreamingChatCompletionClient.start() called twice")
        self._thread.start()

    def _request_body(self) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self._model,
            "messages": self._messages,
            "max_completion_tokens": self._max_completion_tokens,
            "temperature": self._temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        if self._messages and self._messages[-1].get("role") == "assistant":
            body["add_generation_prompt"] = False
            body["continue_final_message"] = True
        return body

    def _append_text(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            self._text_chunks.append(chunk)

    def _worker(self) -> None:
        try:
            self._resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=self._request_body(),
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
                # Strip whitespace and prefix.
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
                usage = ev.get("usage")
                if isinstance(usage, dict):
                    ct = usage.get("completion_tokens")
                    with self._lock:
                        if isinstance(ct, int) and ct > self._completion_tokens:
                            self._completion_tokens = ct
                choices = ev.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                c0 = choices[0]
                if not isinstance(c0, dict):
                    continue
                with self._lock:
                    self._finish_reason = c0.get("finish_reason")
                delta = c0.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    self._append_text(content)
        except Exception as e:
            with self._lock:
                self._error = e
                self._finish_reason = "error"
        finally:
            if self._resp is not None:
                with contextlib.suppress(Exception):
                    self._resp.close()
            with self._lock:
                if self._finish_reason is None and self._cancel.is_set():
                    self._finish_reason = "cancelled"
            self._finished.set()

    def wait_openai_id(self) -> str:
        """Block until the SSE exposes an ``id``, the stream ends, or timeout."""
        deadline = time.time() + Config.OPENAI_STREAM_ID_TIMEOUT_S
        while time.time() < deadline:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError(
                        "streaming chat completion failed before id was seen"
                    ) from self._error
                if self._openai_id:
                    return self._openai_id
            if self._finished.is_set():
                raise RuntimeError(
                    "streaming chat completion finished before id was seen"
                )
            time.sleep(Config.OPENAI_STREAM_ID_INTERVAL_S)
        raise RuntimeError("timeout waiting for streaming OpenAI request id")

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
    def output_text(self) -> str:
        with self._lock:
            return "".join(self._text_chunks)

    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return self._completion_tokens

    @property
    def finish_reason(self) -> str | None:
        with self._lock:
            return self._finish_reason
