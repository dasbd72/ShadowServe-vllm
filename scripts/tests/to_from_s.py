# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging

import torch

from vllm.shadow.models.loader import ShadowModelLoader
from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.engine import AsyncShadow
from vllm.shadow.transfer.kv_transport_common import MemfdTensor
from vllm.shadow.transfer.kvhts_memfd import UdsMemfdKvhtsSenderTransport
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff, KvhtsRequest
from vllm.shadow.transfer.kvstc_memfd import UdsMemfdKvstcReceiverTransport
from vllm.shadow.transfer.tksth_protocol import (
    TksthError,
    TksthFinish,
    TksthMessage,
)
from vllm.shadow.transfer.tksth_uds import UdsTksthReceiverTransport

logger = logging.getLogger("to_from_s.py")


class Context:
    def __init__(self):
        self.engine_args = ShadowEngineArgs(
            model="Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            block_size=16,
        )

        self.model_loader = ShadowModelLoader(self.engine_args.get_model_config())
        self.hf_config = self.model_loader.hf_config

        self.num_layers = self.hf_config.num_hidden_layers
        self.batch_size = 1
        self.shadow_num_blocks = 16
        self.num_kv_heads = self.hf_config.num_key_value_heads
        self.head_dim = self.hf_config.head_dim
        self.block_size = 16
        self.dtype = "bfloat16"

        self.migration_id = 2
        self.request_id = "test-request-id"
        self.kvhts_ipc_path = "/tmp/kvhts.sock"
        self.tksth_ipc_path = "/tmp/tksth.sock"
        self.kvstc_ipc_path = "/tmp/kvstc.sock"

        self.loop: asyncio.AbstractEventLoop | None = None
        self.shadow: AsyncShadow | None = None
        self.shadow_running_ready = asyncio.Event()
        self.cold_task_running_ready = asyncio.Event()

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.shadow = AsyncShadow.from_engine_args(self.engine_args)
        try:
            await self.shadow.shadow_migration_recv(
                self.migration_id, self.kvhts_ipc_path
            )

            hot_task = asyncio.create_task(asyncio.to_thread(self._hot))

            await self.shadow_running_ready.wait()

            cold_task = asyncio.create_task(asyncio.to_thread(self._cold))

            await self.cold_task_running_ready.wait()

            await asyncio.sleep(2)

            migrated_request_ids = await self.shadow.shadow_migration_migrate(
                self.migration_id, self.kvstc_ipc_path
            )
            assert migrated_request_ids == [self.request_id]

            await hot_task
            await cold_task
        finally:
            self.shadow.shutdown()

    def _hot(self):
        self.kvhts_transport = UdsMemfdKvhtsSenderTransport()
        try:
            self.kvhts_transport.connect(self.kvhts_ipc_path)

            handoff = KvhtsHandoff(
                migration_id=self.migration_id,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                shadow_num_blocks=self.shadow_num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                dtype=self.dtype,
                requests=[
                    KvhtsRequest(
                        request_id=self.request_id,
                        prompt_token_ids=[1, 2, 3],
                        output_token_ids=[4],
                        num_computed_tokens=3,
                        block_table=[0],
                        sampling_params={
                            "temperature": 1.0,
                            "max_tokens": 500,
                        },
                    )
                ],
                tksth_ipc_path=self.tksth_ipc_path,
            )

            self.kvhts_transport.send_handoff(handoff)
            for _ in range(int(self.num_layers)):
                layer = MemfdTensor.from_tensor(
                    torch.randn(
                        (
                            2,
                            self.shadow_num_blocks,
                            self.num_kv_heads,
                            self.block_size,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                    )
                )
                self.kvhts_transport.send_layer(layer)
                layer.close()

            self.tksth_transport = UdsTksthReceiverTransport()
            try:
                self.tksth_transport.prepare(self.tksth_ipc_path)
                self.tksth_transport.accept_once()

                pending = {req.request_id for req in handoff.requests}
                messages: list[TksthMessage] = []
                while pending:
                    msg = self.tksth_transport.recv()
                    if not self.shadow_running_ready.is_set():
                        self.loop.call_soon_threadsafe(self.shadow_running_ready.set)

                    messages.append(msg)

                    if isinstance(msg, TksthFinish):
                        pending.discard(msg.request_id)
                    elif isinstance(msg, TksthError):
                        if msg.request_id is not None:
                            pending.discard(msg.request_id)
                        else:
                            pending.clear()
                        raise ValueError(f"TKSTH error: {msg}")

                logger.info("TKSTH messages: %s", messages)
            finally:
                self.tksth_transport.close()
        finally:
            self.kvhts_transport.close()

    def _cold(self):
        self.kvstc_transport = UdsMemfdKvstcReceiverTransport()
        try:
            self.kvstc_transport.prepare(self.kvstc_ipc_path)
            if not self.cold_task_running_ready.is_set():
                self.loop.call_soon_threadsafe(self.cold_task_running_ready.set)
            self.kvstc_transport.accept_once()

            handoff = self.kvstc_transport.recv_handoff()
            assert handoff.migration_id == self.migration_id
            assert handoff.requests[0].request_id == self.request_id

            for _ in range(int(handoff.num_layers)):
                layer = self.kvstc_transport.recv_layer(handoff)
                layer.close()

        finally:
            self.kvstc_transport.close()


def main():
    logging.basicConfig(level=logging.INFO)
    context = Context()
    asyncio.run(context.start())


if __name__ == "__main__":
    main()
