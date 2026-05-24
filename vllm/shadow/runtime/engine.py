# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import logging
import time

from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.core_client import (
    ShadowEngineCoreClient,
    ShadowEngineDeadError,
)
from vllm.shadow.runtime.logger import StatLogger
from vllm.shadow.runtime.utils import cancel_task_threadsafe

__all__ = ("AsyncShadow",)


logger = logging.getLogger("vllm.shadow.runtime.engine")


class AsyncShadow:
    """Async wrapper around a multiprocess :class:`ShadowEngineCore`."""

    def __init__(self, engine_args: ShadowEngineArgs) -> None:
        self.engine_args = engine_args
        self.engine_core: ShadowEngineCoreClient = ShadowEngineCoreClient.make_client(
            engine_args
        )
        self.stat_logger = StatLogger()
        self.output_handler: asyncio.Task | None = None
        try:
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    @classmethod
    def from_engine_args(cls, engine_args: ShadowEngineArgs) -> AsyncShadow:
        return cls(engine_args)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""
        engine_core = getattr(self, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    def _run_output_handler(self):
        """Background loop: pulls from ShadowEngineCore."""

        if self.output_handler is not None:
            return

        engine_core = self.engine_core
        stat_logger = self.stat_logger

        async def output_handler():
            try:
                last_logging_time = time.monotonic()
                while True:
                    outputs = await engine_core.get_output()
                    if stat_logger is not None and outputs.request_ids:
                        stat_logger.record(
                            request_ids=outputs.request_ids,
                            kv_cache_usage=outputs.kv_cache_usage,
                            num_generation_tokens=outputs.num_generation_tokens,
                            preprocess_elapsed=outputs.preprocess_elapsed,
                            forward_elapsed=outputs.forward_elapsed,
                            postprocess_elapsed=outputs.postprocess_elapsed,
                        )
                    now = time.monotonic()
                    if now - last_logging_time > 2.0:
                        last_logging_time = now
                        stat_logger.log()
            except Exception:
                logger.exception("AsyncShadow output_handler failed.")

        self.output_handler = asyncio.create_task(output_handler())

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def shadow_migration_recv(
        self, migration_id: int, kvhts_ipc_path: str
    ) -> None:
        return await self.engine_core.shadow_migration_recv(
            migration_id, kvhts_ipc_path
        )

    async def shadow_migration_migrate(
        self, migration_id: int, kvstc_ipc_path: str
    ) -> list[str]:
        return await self.engine_core.shadow_migration_migrate(
            migration_id, kvstc_ipc_path
        )

    async def shadow_migration_completed(self) -> dict[str, list[int]]:
        return await self.engine_core.shadow_migration_completed()

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return ShadowEngineDeadError()
