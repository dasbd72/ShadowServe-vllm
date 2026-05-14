# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow-local weight loader for CPU model execution.

Loads safetensors (or .bin/.pt fallback) from a local directory or
HuggingFace model id, streaming tensors one-at-a-time in the style of
vLLM's weight iterators — without importing the heavy vLLM runtime.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Generator

import regex as re
import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from vllm.shadow.runtime.config import ShadowConfig

logger = logging.getLogger("vllm.shadow.models.weight_loader")

__all__ = ("ShadowWeightsLoader",)


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key so ``model-00002`` sorts after ``model-00001``."""
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r"(\d+)", os.path.basename(filepath))
    ]


class ShadowWeightsLoader:
    """vLLM-style weight shard iterator for the shadow CPU process."""

    def __init__(self, shadow_config: ShadowConfig) -> None:
        self._config = shadow_config

        self._model_dir = self._resolve_model_dir()

    # ── Model directory resolution ────────────────────────────────────

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def _resolve_model_dir(self) -> str:
        """Resolve ``shadow_config.model`` to a local directory.

        If the model string is already a local directory it is used as-is.
        Otherwise ``huggingface_hub.snapshot_download`` fetches the
        checkpoint (safetensors + JSON config files).
        """
        model = self._config.model

        if self._config.load_format == "serverless_llm":
            return os.path.normpath(os.path.expanduser(model))

        if os.path.isdir(model):
            logger.info("Using local model directory: %s", model)
            return model

        from huggingface_hub import snapshot_download

        logger.info("Downloading model %s ...", model)
        model = snapshot_download(
            repo_id=model,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "*.json",
                "*.bin",
                "*.pt",
            ],
        )

        logger.info("Model resolved to: %s", model)
        return model

    # ── Shard discovery ───────────────────────────────────────────────

    def _discover_safetensors(self, model_dir: str) -> list[str]:
        """Return an ordered list of safetensors shard paths.

        Prefers the ``model.safetensors.index.json`` weight map when
        present; otherwise falls back to a natural-sorted glob.
        """
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map: dict[str, str] = json.load(f)["weight_map"]
            unique_shards: list[str] = list(dict.fromkeys(weight_map.values()))
            paths = [os.path.join(model_dir, s) for s in unique_shards]
            logger.info("Discovered %d safetensors shards via index file", len(paths))
            return paths

        paths = sorted(
            glob.glob(os.path.join(model_dir, "*.safetensors")),
            key=_natural_sort_key,
        )
        if paths:
            logger.info("Discovered %d safetensors shards via glob", len(paths))
        return paths

    def _discover_bin_pt(self, model_dir: str) -> list[str]:
        """Return an ordered list of ``.bin`` / ``.pt`` shard paths."""
        paths = sorted(
            glob.glob(os.path.join(model_dir, "*.bin"))
            + glob.glob(os.path.join(model_dir, "*.pt")),
            key=_natural_sort_key,
        )
        blacklist = {
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        }
        paths = [p for p in paths if os.path.basename(p) not in blacklist]
        if paths:
            logger.info("Discovered %d .bin/.pt shards via glob", len(paths))
        return paths

    # ── Tensor iterators ──────────────────────────────────────────────

    def iter_tensors(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield ``(name, tensor)`` pairs from the checkpoint.

        Tries safetensors first; falls back to ``.bin``/``.pt`` if none
        are found.
        """
        load_format = self._config.load_format

        if load_format == "serverless_llm":
            yield from self._iter_sllm()
            return

        if load_format != "safetensors":
            raise ValueError(
                f"unsupported shadow load_format={load_format!r}; "
                "supported: safetensors, serverless_llm"
            )

        model_dir = self.model_dir
        st_files = self._discover_safetensors(model_dir)

        if st_files:
            yield from self._iter_safetensors(st_files)
            return

        bin_files = self._discover_bin_pt(model_dir)
        if bin_files:
            logger.warning("No safetensors found; falling back to .bin/.pt loading")
            yield from self._iter_bin_pt(bin_files)
            return

        raise FileNotFoundError(
            f"No safetensors or .bin/.pt weight files found in {model_dir}"
        )

    def _iter_sllm(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        from sllm_store.torch import load_dict

        model_id = self._sllm_store_model_id()
        logger.info(
            "SLLM load_dict model_id=%r (local dir=%s)", model_id, self.model_dir
        )
        state = load_dict(model_id, {"": "cpu"})
        for name in list(state.keys()):
            tensor = state.pop(name)
            self._warn_dtype_mismatch(name, tensor.dtype)
            yield name, tensor
            del tensor
        del state

    @staticmethod
    def _remove_storage_prefix(path: str, prefix: str) -> str:
        path = os.path.normpath(path)
        prefix = os.path.normpath(prefix)

        if path == prefix:
            return ""
        if path.startswith(prefix + os.sep):
            return path[len(prefix) :].lstrip(os.sep)
        return path

    def _sllm_store_model_id(self) -> str:
        full = os.path.normpath(os.path.expanduser(self.model_dir))
        if not os.path.isabs(full):
            full = os.path.abspath(full)
        full = os.path.join(full, "rank_0")
        storage_path = os.getenv("STORAGE_PATH", "~/models")
        storage_path = os.path.normpath(os.path.expanduser(storage_path))
        rel = self._remove_storage_prefix(full, storage_path)
        if not rel:
            raise ValueError(
                f"SLLM model directory {full!r} equals STORAGE_PATH {storage_path!r}; "
                "cannot derive store model id."
            )
        if os.path.isabs(rel):
            raise ValueError(
                f"SLLM model directory {full!r} should be under {storage_path!r}."
            )
        return rel

    def _iter_safetensors(
        self,
        shard_files: list[str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for shard_path in tqdm(
            shard_files,
            desc="Loading safetensors shards",
            disable=False,
        ):
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in f.offset_keys():
                    tensor = f.get_tensor(name)
                    yield name, tensor
                    del tensor

    def _iter_bin_pt(
        self,
        shard_files: list[str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for shard_path in tqdm(
            shard_files,
            desc="Loading .bin/.pt shards",
            disable=False,
        ):
            logger.debug("Loading .bin/.pt shard %s", shard_path)
            state = torch.load(shard_path, map_location="cpu", weights_only=True)
            for name in list(state.keys()):
                tensor = state.pop(name)
                self._warn_dtype_mismatch(name, tensor.dtype)
                yield name, tensor
                del tensor
            del state

    # ── Helpers ────────────────────────────────────────────────────────

    def _warn_dtype_mismatch(self, name: str, tensor_dtype: torch.dtype) -> None:
        """Debug use. Warn if the tensor dtype does not match the expected dtype."""
        expected = self._config.dtype
        if not isinstance(expected, torch.dtype):
            return
        if (
            tensor_dtype.is_floating_point
            and expected.is_floating_point
            and tensor_dtype != expected
        ):
            logger.debug(
                "Weight %s has dtype %s (config expects %s)",
                name,
                tensor_dtype,
                expected,
            )
