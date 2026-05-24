# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import multiprocessing
import os
import signal
import tempfile
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Any
from uuid import uuid4

import msgspec
import psutil
import zmq
import zmq.asyncio
from urllib3.util import parse_url

from vllm.shadow.runtime.arg_utils import ShadowEngineArgs

logger = logging.getLogger("vllm.shadow.runtime.utils")


def cancel_task_threadsafe(task: asyncio.Task) -> None:
    if task and not task.done():
        run_in_loop(task.get_loop(), task.cancel)


def in_loop(event_loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() == event_loop
    except RuntimeError:
        return False


def run_in_loop(loop: asyncio.AbstractEventLoop, function: Callable, *args):
    if in_loop(loop):
        function(*args)
    elif not loop.is_closed():
        loop.call_soon_threadsafe(function, *args)


def _rpc_base_path() -> str:
    return os.environ.get("VLLM_RPC_BASE_PATH", tempfile.gettempdir())


def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]) -> None:
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def decode_msgpack_zmq_frames(
    decoder: msgspec.msgpack.Decoder,
    frames: Sequence[bytes | memoryview | zmq.Frame],
) -> Any:
    """Decode a ZMQ multipart payload (v1 ``MsgpackDecoder.decode(data_frames)``).

    The first frame is the msgpack header; additional frames are reserved for
    future zero-copy side buffers when the client adopts ``MsgpackEncoder``.
    """
    if not frames:
        raise ValueError("No payload frames in ZMQ message")
    header = frames[0]
    if isinstance(header, zmq.Frame):
        header = header.buffer
    return decoder.decode(header)


def _is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = parse_url(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if scheme == "tcp" and not all((host, port)):
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def _get_open_zmq_ipc_path() -> str:
    return f"ipc://{_rpc_base_path()}/{uuid4()}"


def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    scheme, host, _ = _split_zmq_path(path)
    if scheme == "tcp" and _is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket."""
    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")
    finally:
        ctx.destroy(linger=linger)


STARTUP_POLL_PERIOD_MS = 10_000
HANDSHAKE_TIMEOUT_MINS = 5
ENGINE_READY_TIMEOUT_S = 300


@dataclass
class ShadowEngineZmqAddresses:
    input_address: str
    output_address: str


@dataclass
class ShadowEngineHandshakeMetadata:
    addresses: ShadowEngineZmqAddresses


def get_shadow_engine_zmq_addresses() -> ShadowEngineZmqAddresses:
    """Allocate ZMQ IPC addresses for shadow engine-client communication."""
    return ShadowEngineZmqAddresses(
        input_address=_get_open_zmq_ipc_path(),
        output_address=_get_open_zmq_ipc_path(),
    )


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _shutdown(procs: list[BaseProcess]) -> None:
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    deadline = time.monotonic() + 5
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


class ShadowCoreProcManager:
    """Manages the background ShadowEngineCore process."""

    def __init__(
        self,
        target_fn: Callable,
        engine_args: ShadowEngineArgs,
        handshake_address: str,
    ):
        context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = [
            context.Process(
                target=target_fn,
                name="ShadowEngineCore",
                kwargs={
                    "engine_args": engine_args,
                    "handshake_address": handshake_address,
                },
            )
        ]
        self._finalizer = weakref.finalize(self, _shutdown, self.processes)
        try:
            for proc in self.processes:
                proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.close()

    def close(self) -> None:
        self._finalizer()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


def _wait_for_shadow_engine_startup(
    handshake_socket: zmq.Socket,
    proc_manager: ShadowCoreProcManager,
    addresses: ShadowEngineZmqAddresses,
) -> None:
    """Wait for the shadow engine core process to complete startup handshake."""
    identity = (0).to_bytes(2, "little")
    state = "HELLO"
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)
    for sentinel in proc_manager.sentinels():
        poller.register(sentinel, zmq.POLLIN)

    while state != "DONE":
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            logger.debug("Waiting for shadow engine core process to start.")
            continue
        if events[0][0] != handshake_socket:
            finished = proc_manager.finished_procs()
            raise RuntimeError(
                "Shadow engine core initialization failed. "
                f"Failed core proc(s): {finished}"
            )

        eng_identity, msg_bytes = handshake_socket.recv_multipart()
        if eng_identity != identity:
            raise RuntimeError(f"Unexpected engine identity: {eng_identity!r}")
        msg = msgspec.msgpack.decode(msg_bytes)
        status = msg["status"]

        if status == "HELLO" and state == "HELLO":
            init_message = msgspec.msgpack.encode(
                ShadowEngineHandshakeMetadata(addresses=addresses)
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            state = "READY"
        elif status == "READY" and state == "READY":
            state = "DONE"
        else:
            raise RuntimeError(
                f"Unexpected handshake message status={status!r} state={state!r}"
            )


@contextlib.contextmanager
def launch_shadow_core_engine(
    engine_args: ShadowEngineArgs,
    addresses: ShadowEngineZmqAddresses,
) -> Iterator[ShadowCoreProcManager]:
    """Launch the shadow engine core process and complete startup handshake."""
    from vllm.shadow.runtime.core import ShadowEngineCoreProc

    handshake_address = _get_open_zmq_ipc_path()
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        proc_manager = ShadowCoreProcManager(
            ShadowEngineCoreProc.run_engine_core,
            engine_args=engine_args,
            handshake_address=handshake_address,
        )
        try:
            yield proc_manager
        finally:
            _wait_for_shadow_engine_startup(handshake_socket, proc_manager, addresses)
