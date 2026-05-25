# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cold_start_interference benchmarks sequentially; one log per run."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("run_suite.py")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = ROOT / "logs"
BENCHMARK_SCRIPT = ROOT / "scripts" / "shadow_migration" / "cold_start_interference.py"

CONCURRENT_CPUS = (8, 16, 32, 64, 128)
MEM_BW_CPUS = (1, 2, 4, 8, 16, 32)

MODEL_SLUGS: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-8B": "qwen3-8b",
}


class Context:
    def __init__(self, logs_dir: Path, models: list[str]) -> None:
        self.logs_dir = logs_dir
        self.models = models
        self.script = BENCHMARK_SCRIPT

    def run_all(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for log_name, argv in self._suite_runs():
            self._run_one(log_name, argv)
        logger.info("All runs complete. Logs in %s/", self.logs_dir)

    def _suite_runs(self) -> list[tuple[str, list[str]]]:
        runs: list[tuple[str, list[str]]] = []

        for model in self.models:
            slug = MODEL_SLUGS[model]
            runs.append(
                (
                    f"cold_start_baseline_{slug}.log",
                    ["--mode", "baseline", "--model", model],
                )
            )

        for cpus in CONCURRENT_CPUS:
            for model in self.models:
                slug = MODEL_SLUGS[model]
                runs.append(
                    (
                        f"cold_start_concurrent_{slug}_cpus{cpus}.log",
                        [
                            "--mode",
                            "concurrent",
                            "--workload-cpus",
                            str(cpus),
                            "--model",
                            model,
                            "--num-requests",
                            "4",
                            "--input-len",
                            "1024",
                        ],
                    )
                )

        for cpus in MEM_BW_CPUS:
            for model in self.models:
                slug = MODEL_SLUGS[model]
                runs.append(
                    (
                        f"cold_start_mem_bw_{slug}_cpus{cpus}.log",
                        [
                            "--mode",
                            "mem_bw",
                            "--workload-cpus",
                            str(cpus),
                            "--model",
                            model,
                        ],
                    )
                )

        return runs

    def _run_one(self, log_name: str, argv: list[str]) -> None:
        log_path = self.logs_dir / log_name
        cmd = ["python", str(self.script), *argv]
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s starting -> %s", started, log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s finished -> %s", finished, log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for per-run log files (default: <repo>/logs)",
    )
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=sorted(MODEL_SLUGS),
        metavar="MODEL",
        help=(
            "Model to benchmark (repeatable). "
            f"Choices: {', '.join(sorted(MODEL_SLUGS))}. "
            "Default: all models."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    models = args.models or sorted(MODEL_SLUGS)
    context = Context(logs_dir=args.logs_dir, models=models)
    try:
        context.run_all()
    except subprocess.CalledProcessError as exc:
        logger.error("benchmark failed (exit %s): %s", exc.returncode, exc.cmd)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
