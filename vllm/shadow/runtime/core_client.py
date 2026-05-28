# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import uuid
import weakref
from dataclasses import dataclass
from threading import Thread
from typing import Any

import msgspec
import zmq
import zmq.asyncio

from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.core import (
    ENGINE_CORE_IDENTITY,
    ShadowEngineCoreOutputs,
    ShadowEngineCoreProc,
    ShadowEngineCoreRequestType,
)
from vllm.shadow.runtime.utils import (
    ENGINE_READY_TIMEOUT_S,
    ShadowCoreProcManager,
    close_sockets,
    get_shadow_engine_zmq_addresses,
    in_loop,
    launch_shadow_core_engine,
    make_zmq_socket,
)

logger = logging.getLogger("vllm.shadow.runtime.core_client")


class ShadowEngineDeadError(RuntimeError):
    """Raised when the background shadow engine core process has died."""


@dataclass
class ShadowBackgroundResources:
    """Finalizer-owned ZMQ and process resources for clean shutdown."""

    ctx: zmq.Context
    engine_manager: ShadowCoreProcManager | None = None
    output_socket: zmq.asyncio.Socket | None = None
    input_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None

    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.close()

        # Async case.
        loop = self.output_queue_task._loop if self.output_queue_task else None
        sockets = (
            self.output_socket,
            self.input_socket,
        )
        tasks = (self.output_queue_task,)

        def close_sockets_and_tasks():
            close_sockets(sockets)
            for task in tasks:
                if task is not None and not task.done():
                    with contextlib.suppress(Exception):
                        task.cancel()

        if loop is not None:
            if in_loop(loop):
                close_sockets_and_tasks()
            elif not loop.is_closed():
                loop.call_soon_threadsafe(close_sockets_and_tasks)
        else:
            # Loop has been closed, try to clean up directly.
            del tasks
            del close_sockets_and_tasks
            close_sockets(sockets)
            del self.output_queue_task

    def validate_alive(self, frames):
        if (
            len(frames) == 1
            and frames[0].buffer == ShadowEngineCoreProc.ENGINE_CORE_DEAD
        ):
            self.engine_dead = True
            raise ShadowEngineDeadError()


def _process_utility_output(
    output: dict[str, Any], utility_results: dict[int, asyncio.Future[Any]]
) -> None:
    call_id = output["call_id"]
    future = utility_results.pop(call_id)
    failure_message = output.get("failure_message")
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            future.set_result(output.get("result"))
    except asyncio.InvalidStateError:
        if failure_message is not None:
            logger.error("Cancelled shadow utility call failed: %s", failure_message)


class ShadowEngineCoreClient:
    """Async ZMQ client for a :class:`ShadowEngineCore` in a background process."""

    @staticmethod
    def make_client(engine_args: ShadowEngineArgs) -> ShadowEngineCoreClient:
        return ShadowEngineCoreClient(engine_args)

    def __init__(self, engine_args: ShadowEngineArgs):
        self.engine_args = engine_args
        # Serialization setup.
        self.encoder = msgspec.msgpack.Encoder()

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx)

        self.resources = ShadowBackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        self.core_engine = ENGINE_CORE_IDENTITY
        try:
            addresses = get_shadow_engine_zmq_addresses()
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, addresses.input_address, zmq.ROUTER, bind=True
            )
            self.resources.output_socket = make_zmq_socket(
                self.ctx, addresses.output_address, zmq.PULL
            )

            with launch_shadow_core_engine(engine_args, addresses) as engine_manager:
                self.resources.engine_manager = engine_manager

            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            if not sync_input_socket.poll(timeout=ENGINE_READY_TIMEOUT_S * 1000):
                raise TimeoutError(
                    f"Timed out waiting for shadow engine core process to start "
                    f"({ENGINE_READY_TIMEOUT_S}s)."
                )
            identity, _ = sync_input_socket.recv_multipart()
            if identity != self.core_engine:
                raise RuntimeError(
                    f"Unexpected engine identity during startup: {identity!r}"
                )

            self.utility_results: dict[int, asyncio.Future[Any]] = {}

            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

        self.outputs_queue = asyncio.Queue[ShadowEngineCoreOutputs | Exception]()
        try:
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def shutdown(self):
        self._finalizer()

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise ShadowEngineDeadError()

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            # No engine processes to monitor
            return

        engine_processes = engine_manager.processes
        self_ref = weakref.ref(self)

        def monitor_engine_core():
            sentinels = [proc.sentinel for proc in engine_processes]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            proc_name = next(
                proc.name for proc in engine_processes if proc.sentinel == died[0]
            )
            logger.error(
                "Shadow engine core proc %s died unexpectedly, shutting down client.",
                proc_name,
            )
            _self.shutdown()

        Thread(
            target=monitor_engine_core,
            daemon=True,
            name="ShadowMPClientEngineMonitor",
        ).start()

    def _ensure_output_queue_task(self) -> None:
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_socket = resources.output_socket
        assert isinstance(output_socket, zmq.asyncio.Socket)

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs = ShadowEngineCoreOutputs.decode(frames)
                    if outputs.utility_output is not None:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(ShadowEngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="ShadowEngineCoreOutputQueueTask"
        )

    async def get_output(self) -> ShadowEngineCoreOutputs:
        self._ensure_output_queue_task()
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            if self.resources.engine_dead:
                raise ShadowEngineDeadError() from None
            raise outputs
        return outputs

    def _send_input(self, request_type: ShadowEngineCoreRequestType, request: Any):
        self.ensure_alive()
        msg = (self.core_engine, request_type.value, self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    async def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        self._send_input(ShadowEngineCoreRequestType.UTILITY, (call_id, method, args))
        self._ensure_output_queue_task()
        return await future

    async def shadow_migration_recv(
        self, migration_id: int, kvhts_ipc_path: str
    ) -> None:
        return await self.call_utility(
            "shadow_migration_recv", migration_id, kvhts_ipc_path
        )

    async def shadow_migration_migrate(
        self, migration_id: int, kvstc_ipc_path: str
    ) -> list[str]:
        return await self.call_utility(
            "shadow_migration_migrate", migration_id, kvstc_ipc_path
        )

    async def shadow_migration_completed(self) -> dict[str, list[int]]:
        return await self.call_utility("shadow_migration_completed")
