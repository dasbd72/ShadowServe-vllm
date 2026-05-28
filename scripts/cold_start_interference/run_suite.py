# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cold_start_interference benchmarks sequentially; one log per run."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("scripts/cold_start_interference/run_suite.py")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = ROOT / "logs"
BENCHMARK_SCRIPT = Path(__file__).parent / "run.py"


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    mode: str
    log: Path
    workload_cpus: int | None = None
    num_requests: int | None = None
    input_len: int | None = None
    num_sessions: int | None = None

    def argv(self) -> list[str]:
        args = ["--mode", self.mode, "--model", self.model]
        if self.workload_cpus is not None:
            args.extend(["--workload-cpus", str(self.workload_cpus)])
        if self.num_requests is not None:
            args.extend(["--num-requests", str(self.num_requests)])
        if self.input_len is not None:
            args.extend(["--input-len", str(self.input_len)])
        if self.num_sessions is not None:
            args.extend(["--num-sessions", str(self.num_sessions)])
        return args


class Context:
    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.script = BENCHMARK_SCRIPT

    def run(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for scenario in self._suite_scenarios():
            self._run_scenario(scenario)
        logger.info("All runs complete. Logs in %s/", self.logs_dir)

    def _suite_scenarios(self) -> list[Scenario]:
        scenarios: list[Scenario] = []

        model_slugs: dict[str, str] = {
            "Qwen/Qwen3-0.6B": "qwen3-0.6b",
            "Qwen/Qwen3-4B": "qwen3-4b",
            "Qwen/Qwen3-8B": "qwen3-8b",
        }
        concurrent_cpus = (
            8,
            16,
            32,
            64,
            128,
        )
        mem_bw_cpus = (
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
        )

        for model, slug in model_slugs.items():
            scenarios.append(
                Scenario(
                    name=f"baseline-{slug}",
                    model=model,
                    mode="baseline",
                    log=self.logs_dir / f"cold_start_{slug}_baseline.log",
                )
            )

        for cpus in concurrent_cpus:
            for model, slug in model_slugs.items():
                scenarios.append(
                    Scenario(
                        name=f"concurrent-{slug}-cpus{cpus}",
                        model=model,
                        mode="concurrent",
                        log=self.logs_dir
                        / f"cold_start_{slug}_concurrent_cpus{cpus}.log",
                        workload_cpus=cpus,
                        num_requests=2,
                        input_len=1024,
                        num_sessions=2,
                    )
                )

        for cpus in mem_bw_cpus:
            for model, slug in model_slugs.items():
                scenarios.append(
                    Scenario(
                        name=f"mem_bw-{slug}-cpus{cpus}",
                        model=model,
                        mode="mem_bw",
                        log=self.logs_dir / f"cold_start_{slug}_mem_bw_cpus{cpus}.log",
                        workload_cpus=cpus,
                    )
                )

        return scenarios

    def _run_scenario(self, scenario: Scenario) -> None:
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info(
            "==> %s starting scenario %s -> %s",
            started,
            scenario.name,
            scenario.log,
        )
        scenario.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(self.script), *scenario.argv()]
        with scenario.log.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info(
            "==> %s finished scenario %s -> %s",
            finished,
            scenario.name,
            scenario.log,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for per-run log files (default: <repo>/logs)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    context = Context(logs_dir=args.logs_dir)
    try:
        context.run()
    except subprocess.CalledProcessError as exc:
        logger.error("benchmark failed (exit %s): %s", exc.returncode, exc.cmd)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
