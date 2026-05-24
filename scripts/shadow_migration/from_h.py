# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging

from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.shadow.transfer.kvhts_memfd import UdsMemfdKvhtsReceiverTransport
from vllm.shadow.transfer.tksth_protocol import TksthFinish, TksthTokenDelta
from vllm.shadow.transfer.tksth_uds import UdsTksthSenderTransport
from vllm.v1.engine.output_processor import RequestOutputCollector

logger = logging.getLogger("from_h.py")


class Context:
    def __init__(self):
        self.engine_args = AsyncEngineArgs(
            model="Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            block_size=16,
            shadow_sender_enabled=True,
            shadow_additional_blocks_per_request=1,
            shadow_tksth_ipc_prefix="/tmp/tksth",
        )

        self.migration_id = 1
        self.kvhts_ipc_path = "/tmp/kvhts.sock"

        self.loop: asyncio.AbstractEventLoop | None = None
        self.llm: AsyncLLMEngine | None = None
        self.shadow_running_ready = asyncio.Event()
        self.request_running_ready = asyncio.Event()

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.llm = AsyncLLMEngine.from_engine_args(self.engine_args)
        try:
            gen_task = asyncio.create_task(self._generate())

            await self.request_running_ready.wait()

            shadow_task = asyncio.create_task(asyncio.to_thread(self._shadow))

            await self.shadow_running_ready.wait()

            migrated_requests = await self.llm.shadow_migration_migrate(
                self.migration_id, self.kvhts_ipc_path, 1
            )
            assert len(migrated_requests) == 1

            await shadow_task
            await gen_task
        finally:
            self.llm.shutdown()

    def _shadow(self) -> None:
        receiver = UdsMemfdKvhtsReceiverTransport()
        try:
            receiver.prepare(self.kvhts_ipc_path)
            if not self.shadow_running_ready.is_set():
                self.loop.call_soon_threadsafe(self.shadow_running_ready.set)

            receiver.accept_once()
            handoff = receiver.recv_handoff()
            assert handoff.migration_id == self.migration_id

            for _ in range(int(handoff.num_layers)):
                receiver.recv_layer(handoff)

            request_id = handoff.requests[0].request_id
            sender = UdsTksthSenderTransport()
            try:
                sender.connect(handoff.tksth_ipc_path)
                sender.send(
                    TksthTokenDelta(
                        migration_id=self.migration_id,
                        request_id=request_id,
                        token_ids=[1, 2, 3],
                    )
                )
                sender.send(
                    TksthFinish(
                        migration_id=self.migration_id,
                        request_id=request_id,
                        finish_reason="stop",
                    )
                )
            finally:
                sender.close()
        finally:
            receiver.close()

    async def _generate(self) -> None:
        q: RequestOutputCollector | None = None
        try:
            q = await self.llm.add_request(
                "test-request-id", "Hello, world", SamplingParams(max_tokens=10)
            )

            final_output: RequestOutput | None = None
            finished = False
            while not finished:
                out = q.get_nowait() or await q.get()
                if not self.request_running_ready.is_set():
                    self.request_running_ready.set()

                assert isinstance(out, RequestOutput)
                final_output = out
                finished = out.finished

            logger.info("Final output: %s", final_output)
        finally:
            if q is not None:
                q.close()


def main():
    logging.basicConfig(level=logging.INFO)
    context = Context()
    asyncio.run(context.start())


if __name__ == "__main__":
    main()
