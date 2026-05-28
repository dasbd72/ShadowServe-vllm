# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging

import torch

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.shadow.models.loader import ShadowModelLoader
from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.transfer.kv_transport_common import MemfdTensor
from vllm.shadow.transfer.kvstc_memfd import UdsMemfdKvstcSenderTransport
from vllm.shadow.transfer.kvstc_protocol import KvstcHandoff, KvstcRequest

logger = logging.getLogger("to_c.py")


class Context:
    def __init__(self):
        self.engine_args = AsyncEngineArgs(
            model="Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            block_size=16,
            shadow_receiver_enabled=True,
        )

        self.model_loader = ShadowModelLoader(
            ShadowEngineArgs(
                model="Qwen/Qwen3-0.6B",
                dtype="bfloat16",
                block_size=16,
            ).get_model_config()
        )
        self.hf_config = self.model_loader.hf_config

        self.num_layers = self.hf_config.num_hidden_layers
        self.batch_size = 1
        self.shadow_num_blocks = 16
        self.num_kv_heads = self.hf_config.num_key_value_heads
        self.head_dim = self.hf_config.head_dim
        self.block_size = 16
        self.dtype = "bfloat16"

        self.migration_id = 3
        self.request_id = "test-request-id"
        self.kvstc_ipc_path = "/tmp/kvstc.sock"

        self.loop: asyncio.AbstractEventLoop | None = None
        self.llm: AsyncLLMEngine | None = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.llm = AsyncLLMEngine.from_engine_args(self.engine_args)
        try:
            await self.llm.shadow_migration_recv(self.migration_id, self.kvstc_ipc_path)

            shadow_task = asyncio.create_task(asyncio.to_thread(self._shadow))

            await shadow_task
        finally:
            self.llm.shutdown()

    def _shadow(self) -> None:
        sender = UdsMemfdKvstcSenderTransport()
        try:
            sender.connect(self.kvstc_ipc_path)

            handoff = KvstcHandoff(
                migration_id=self.migration_id,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                shadow_num_blocks=self.shadow_num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                dtype=self.dtype,
                requests=[
                    KvstcRequest(
                        request_id=self.request_id,
                        token_ids=[1, 2, 3],
                        block_table=[0],
                    )
                ],
            )
            sender.send_handoff(handoff)
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
                sender.send_layer(layer)
                layer.close()
        finally:
            sender.close()


def main():
    logging.basicConfig(level=logging.INFO)
    context = Context()
    asyncio.run(context.start())


if __name__ == "__main__":
    main()
