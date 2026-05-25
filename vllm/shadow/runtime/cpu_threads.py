# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenMP/Torch thread binding and NUMA memory policy for shadow CPU decode.

Shadow defaults to no CPU/NUMA binding. vLLM's CPU executor uses
``VLLM_CPU_OMP_THREADS_BIND=auto``; inheriting that here regresses decode
(weights are loaded after init, strict MEMBIND fights Ray cgroup placement).
Set an explicit CPU list (e.g. ``0-7``) to opt in to ``init_cpu_threads_env``.
"""

from __future__ import annotations

import glob
import importlib
import json
import logging
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

from vllm.platforms.interface import CpuArchEnum

logger = logging.getLogger("vllm.shadow.runtime.cpu_threads")

_CPU_VISIBLE_MEMORY_NODES_ENV = "CPU_VISIBLE_MEMORY_NODES"
_RESERVED_CPU_ENV = "VLLM_CPU_NUM_OF_RESERVED_CPU"
_OMP_THREADS_BIND_ENV = "VLLM_CPU_OMP_THREADS_BIND"


@dataclass(slots=True)
class _LogicalCPUInfo:
    id: int
    physical_core: int
    numa_node: int

    @classmethod
    def _int(cls, value: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _json_decoder(obj_dict: dict) -> _LogicalCPUInfo | dict:
        cpu_id = obj_dict.get("cpu")
        physical_core = obj_dict.get("core")
        numa_node = obj_dict.get("node")
        if cpu_id is None or physical_core is None or numa_node is None:
            return obj_dict
        return _LogicalCPUInfo(
            id=_LogicalCPUInfo._int(cpu_id),
            physical_core=_LogicalCPUInfo._int(physical_core),
            numa_node=_LogicalCPUInfo._int(numa_node),
        )


def _get_max_threads(pid: int = 0) -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(pid))
    if platform.system() == "Darwin":
        return os.cpu_count() or 1
    raise NotImplementedError("Unsupported OS")


def _cpu_arch() -> CpuArchEnum:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "i386", "i686"):
        return CpuArchEnum.X86
    if machine.startswith("arm") or machine.startswith("aarch"):
        return CpuArchEnum.ARM
    if machine.startswith("ppc"):
        return CpuArchEnum.POWERPC
    if machine == "s390x":
        return CpuArchEnum.S390X
    return CpuArchEnum.OTHER


def _cpu_selector_for_arch() -> Callable[
    [list[_LogicalCPUInfo]], list[_LogicalCPUInfo]
]:
    arch = _cpu_arch()
    if arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
        return lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
    if arch == CpuArchEnum.X86:
        return lambda cpus: cpus[-1:]
    return lambda cpus: cpus


def _allowed_logical_cpus() -> tuple[list[int], list[_LogicalCPUInfo]]:
    if platform.system() != "Linux":
        return [], []

    lscpu_output = subprocess.check_output(
        "lscpu -J -e=CPU,CORE,NODE", shell=True, text=True
    )
    lscpu_output = re.sub(r'"node":\s*-\s*(,|\n)', r'"node": 0\1', lscpu_output)
    logical_cpu_list: list[_LogicalCPUInfo] = json.loads(
        lscpu_output, object_hook=_LogicalCPUInfo._json_decoder
    )["cpus"]

    logical_cpu_list = [
        x
        for x in logical_cpu_list
        if isinstance(x, _LogicalCPUInfo)
        and -1 not in (x.id, x.physical_core, x.numa_node)
    ]

    if not hasattr(os, "sched_getaffinity"):
        return [], logical_cpu_list
    allowed_cpu_id_list = os.sched_getaffinity(0)
    logical_cpu_list = [x for x in logical_cpu_list if x.id in allowed_cpu_id_list]

    allowed_numa_nodes = sorted({x.numa_node for x in logical_cpu_list})
    if (
        _CPU_VISIBLE_MEMORY_NODES_ENV in os.environ
        and os.environ[_CPU_VISIBLE_MEMORY_NODES_ENV] != ""
    ):
        raw_nodes = os.environ[_CPU_VISIBLE_MEMORY_NODES_ENV]
        visible_nodes = [int(s) for s in raw_nodes.split(",")]
        allowed_numa_nodes = [
            x for x in sorted(set(visible_nodes)) if x in allowed_numa_nodes
        ]

    return allowed_numa_nodes, logical_cpu_list


def _reserve_cpu_num() -> int:
    """Mirror ``CPUWorker._get_autobind_cpu_ids`` (shadow is always rank 0)."""
    raw = os.environ.get(_RESERVED_CPU_ENV)
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", _RESERVED_CPU_ENV, raw)
            return 0
    return 0


def _auto_bind_cpu_ids() -> str | None:
    allowed_numa_nodes, logical_cpu_list = _allowed_logical_cpus()
    if not logical_cpu_list:
        return None

    if allowed_numa_nodes:
        selected_numa_node = allowed_numa_nodes[0]
        logical_cpu_list = [
            x for x in logical_cpu_list if x.numa_node == selected_numa_node
        ]

    cpu_selector = _cpu_selector_for_arch()
    core_to_cpus: dict[int, list[_LogicalCPUInfo]] = {}
    for cpu_info in logical_cpu_list:
        core_to_cpus.setdefault(cpu_info.physical_core, []).append(cpu_info)

    selected: list[_LogicalCPUInfo] = []
    for cpu_list in core_to_cpus.values():
        cpu_list = sorted(cpu_list, key=lambda x: x.id)
        selected.extend(cpu_selector(cpu_list))
    selected = sorted(selected, key=lambda x: x.id)

    reserve_cpu_num = _reserve_cpu_num()
    if reserve_cpu_num != 0:
        if len(selected) <= reserve_cpu_num:
            logger.warning(
                "%s (%d) >= available CPUs (%d); no reservation applied",
                _RESERVED_CPU_ENV,
                reserve_cpu_num,
                len(selected),
            )
        else:
            selected = selected[:-reserve_cpu_num]

    if not selected:
        return None

    logger.info(
        "shadow auto thread-binding list (id, physical core): %s",
        [(x.id, x.physical_core) for x in selected],
    )
    return ",".join(str(x.id) for x in selected)


def _shadow_omp_threads_bind() -> str:
    """Effective bind mode for shadow (default nobind; do not inherit vLLM ``auto``)."""
    raw = os.environ.get(_OMP_THREADS_BIND_ENV)
    if raw is None:
        return "nobind"
    bind_env = raw.strip()
    if bind_env in ("", "auto"):
        return "nobind"
    return bind_env


def _resolve_cpu_bind_ids() -> str | None:
    bind_env = _shadow_omp_threads_bind()
    if bind_env == "nobind":
        return None
    return bind_env.split("|", maxsplit=1)[0]


def _apply_cpu_platform_env() -> None:
    """Mirror ``CpuPlatform.check_and_update_config`` CPU executor env vars."""
    os.environ["NUMEXPR_MAX_THREADS"] = str(_get_max_threads())

    if _shadow_omp_threads_bind() != "nobind":
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
    else:
        logger.info("Disabling binding processes to CPU cores...")

    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    ld_preload_str = os.getenv("LD_PRELOAD", "")
    if "libiomp5.so" in ld_preload_str:
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_TPAUSE"] = "0"
        os.environ["KMP_FORKJOIN_BARRIER_PATTERN"] = "dist,dist"
        os.environ["KMP_PLAIN_BARRIER_PATTERN"] = "dist,dist"
        os.environ["KMP_REDUCTION_BARRIER_PATTERN"] = "dist,dist"

    if (
        platform.system() == "Linux"
        and _cpu_arch() in (CpuArchEnum.ARM, CpuArchEnum.POWERPC)
        and not ("libomp" in ld_preload_str or "libgomp" in ld_preload_str)
    ):
        torch_pkg = os.path.dirname(torch.__file__)
        site_root = os.path.dirname(torch_pkg)
        torch_libs_paths = [
            os.path.join(site_root, "torch.libs"),
            os.path.join(torch_pkg, "lib"),
        ]
        pytorch_libgomp_so_candidates: list[str] = []
        for torch_libs in torch_libs_paths:
            pytorch_libgomp_so_candidates.extend(
                glob.glob(os.path.join(torch_libs, "libgomp*.so*"))
            )
        if pytorch_libgomp_so_candidates:
            pytorch_libgomp_so = pytorch_libgomp_so_candidates[0]
            if ld_preload_str:
                ld_preload_str += ":"
            ld_preload_str += pytorch_libgomp_so
            os.environ["LD_PRELOAD"] = ld_preload_str


def _find_init_cpu_threads_env() -> Any | None:
    for lib in ("_C", "_cpu_ops"):
        ns = getattr(torch.ops, lib, None)
        if ns is not None and hasattr(ns, "init_cpu_threads_env"):
            return ns.init_cpu_threads_env

    if torch.version.cuda is not None or torch.version.hip is not None:
        candidates = ("vllm._cpu_C", "vllm._cpu_C_AVX2", "vllm._C", "vllm._C_AVX2")
    else:
        candidates = ("vllm._C", "vllm._C_AVX2", "vllm._cpu_C", "vllm._cpu_C_AVX2")

    for mod in candidates:
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
        for lib in ("_C", "_cpu_ops"):
            ns = getattr(torch.ops, lib, None)
            if ns is not None and hasattr(ns, "init_cpu_threads_env"):
                return ns.init_cpu_threads_env
    return None


def init_shadow_cpu_threads_env() -> None:
    """Bind worker threads and NUMA memory like vLLM ``CPUWorker.init_device``.

    Applies the same CPU-platform environment variables as ``CpuPlatform``, then
    calls ``init_cpu_threads_env`` when binding is enabled. Falls back to
    ``torch.set_num_threads`` from ``OMP_NUM_THREADS`` when binding is disabled
    or unavailable.
    """
    _apply_cpu_platform_env()

    cpu_ids = _resolve_cpu_bind_ids()
    init_env = _find_init_cpu_threads_env()

    if cpu_ids is not None and init_env is not None:
        try:
            report = init_env(cpu_ids)
        except Exception:
            logger.exception(
                "init_cpu_threads_env(%r) failed; falling back to OMP_NUM_THREADS",
                cpu_ids,
            )
        else:
            if report:
                logger.info("%s", report.rstrip())
            os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
            return

    if cpu_ids is not None and init_env is None:
        logger.warning(
            "init_cpu_threads_env unavailable; NUMA-local memory binding skipped"
        )

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    logger.info("OMP_NUM_THREADS=%r", omp_num_threads)
    if omp_num_threads is None:
        return
    try:
        n = int(omp_num_threads)
    except ValueError:
        logger.warning(
            "Ignoring invalid OMP_NUM_THREADS=%r for torch.set_num_threads",
            omp_num_threads,
        )
        return
    logger.info("Setting torch threads to %d (no CPU/NUMA bind)", n)
    torch.set_num_threads(n)
