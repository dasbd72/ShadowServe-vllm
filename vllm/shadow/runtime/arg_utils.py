# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import dataclasses
from dataclasses import dataclass

from vllm.shadow.runtime.config import ShadowModelConfig


@dataclass(slots=True)
class ShadowEngineArgs:
    log_level: str = "INFO"

    model: str = "Qwen/Qwen3-0.6B"
    load_format: str = "safetensors"
    dtype: str = "auto"
    max_model_len: int | None = None
    block_size: int = 16

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging verbosity.",
        )
        parser.add_argument(
            "model",
            default="Qwen/Qwen3-0.6B",
            help="Same model id/path as ``vllm serve`` "
            "(loads HF config + safetensors).",
        )
        parser.add_argument(
            "--load-format", default="safetensors", help="Load format for the model."
        )
        parser.add_argument(
            "--dtype",
            default="auto",
            help="Weight/KV dtype alignment with the GPU server (e.g. bfloat16).",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Mirrors ``--max-model-len`` on the GPU process. Use -1 for auto.",
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="KV block size; must match the hot pod ``--block-size``.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)},
        )
        return engine_args

    def get_model_config(self) -> ShadowModelConfig:
        return ShadowModelConfig(
            model=self.model,
            load_format=self.load_format,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            block_size=self.block_size,
        )
