# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import collections
import gc
import logging
import os

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_default_torch_dtype

logger = logging.getLogger("vllm.model_executor.model_loader.sllm_loader")


class ServerlessLLMLoader(BaseModelLoader):
    # DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
    _OPTIONAL_BUFFER_SUFFIXES = (
        "._k_scale",
        "._v_scale",
        "._q_scale",
        "._prob_scale",
    )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        # self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result

    @staticmethod
    def _remove_storage_prefix(path: str, prefix: str) -> str:
        path = os.path.normpath(path)
        prefix = os.path.normpath(prefix)

        if path == prefix:
            return ""
        if path.startswith(prefix + os.sep):
            return path[len(prefix) :].lstrip(os.sep)
        return path

    @classmethod
    def _is_optional_buffer(cls, name: str) -> bool:
        return name.endswith(cls._OPTIONAL_BUFFER_SUFFIXES)

    @staticmethod
    def _cast_tensor(
        tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if tensor.is_floating_point() or tensor.is_complex():
            if tensor.device == device:
                if tensor.dtype != dtype:
                    raise ValueError(
                        "SLLM checkpoint dtype does not match vLLM dtype: "
                        f"checkpoint has {tensor.dtype}, vLLM requested "
                        f"{dtype}. Re-run deploy with matching "
                        "`backend_config.torch_dtype` or re-save the model "
                        "with the requested dtype."
                    )
                return tensor
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    def _get_sllm_model_path(self, model_path: str) -> str:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        local_model_path = os.path.join(model_path, f"rank_{rank}")

        # vLLM needs a local model path to read model config but
        # ServerlessLLM Store requires a global model path as the model ID
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        return self._remove_storage_prefix(local_model_path, storage_path)

    def _load_sllm_state_dict(self, model_path: str) -> dict[str, torch.Tensor]:
        from sllm_store.torch import load_dict

        device_id = torch.cuda.current_device()
        return load_dict(model_path, {"": device_id})

    def _load_parameters(
        self,
        model: nn.Module,
        expected_state: dict[str, torch.Tensor],
        sllm_state: dict[str, torch.Tensor],
    ) -> None:
        for name, param in model.named_parameters(recurse=True):
            if name not in expected_state:
                continue

            tensor = sllm_state.get(name)
            if tensor is None:
                continue

            param.data = self._cast_tensor(
                tensor,
                device=param.device,
                dtype=param.dtype,
            )
            expected_state.pop(name)

    def _load_buffers(
        self,
        model: nn.Module,
        expected_state: dict[str, torch.Tensor],
        sllm_state: dict[str, torch.Tensor],
        target_device: torch.device,
    ) -> None:
        for name, buffer in model.named_buffers(recurse=True):
            if name not in expected_state:
                continue

            tensor = sllm_state.get(name)
            if tensor is None:
                if self._is_optional_buffer(name):
                    expected_state.pop(name)
                continue

            buffer.data = self._cast_tensor(
                tensor,
                device=target_device,
                dtype=buffer.dtype,
            ).reshape_as(buffer.data)
            expected_state.pop(name)

    @staticmethod
    def _move_remaining_buffers_to_device(
        model: nn.Module, target_device: torch.device
    ) -> None:
        for _, buffer in model.named_buffers(recurse=True):
            if buffer.device != target_device:
                buffer.data = buffer.data.to(target_device)

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        model_path = model_config.model
        if not os.path.isdir(model_path):
            raise ValueError(
                f"ServerlessLLMLoader requires a local model path, got {model_path!r}"
            )

        sllm_model_path = self._get_sllm_model_path(model_path)
        load_device = (
            vllm_config.device_config.device
            if vllm_config.load_config.device is None
            else vllm_config.load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device("cpu"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                model = model.eval()

            expected_state = self._filter_subtensors(model.state_dict())

            # Release CPU parameter storage before pulling tensors from SLLM
            # store. Buffers stay initialized so vLLM runtime buffers can keep
            # their default values when absent from older checkpoints.
            for name, param in model.named_parameters(recurse=True):
                if name in expected_state:
                    param.data = torch.empty(1, device=target_device)
            gc.collect()

            sllm_state_dict = self._load_sllm_state_dict(
                sllm_model_path,
            )
            self._load_parameters(model, expected_state, sllm_state_dict)
            self._load_buffers(
                model,
                expected_state,
                sllm_state_dict,
                target_device,
            )

            if expected_state:
                raise ValueError(
                    f"Missing keys {tuple(expected_state)} in loaded state!"
                )

            self._move_remaining_buffers_to_device(model, target_device)
            process_weights_after_loading(
                model,
                model_config,
                target_device,
            )

        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from sllm_store.torch import save_dict

        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        state_dict = ServerlessLLMLoader._filter_subtensors(model.state_dict())
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not ServerlessLLMLoader._is_optional_buffer(k)
        }

        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict(state_dict, save_path)
