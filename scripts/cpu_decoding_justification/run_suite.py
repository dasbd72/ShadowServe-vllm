# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cpu_decoding_justification benchmarks sequentially; one log per run."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("scripts.cpu_decoding_justification.run_suite")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = ROOT / "logs" / "cpu_decoding_justification"
BENCHMARK_SCRIPT = Path(__file__).parent / "run.py"


# Qwen3-4B defaults to 40960 context; cap KV cache for single-GPU benches.
MODEL_MAX_MODEL_LEN: dict[str, int] = {
    "Qwen/Qwen3-4B": 20480,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    batch_size: int
    input_len: int
    log: Path
    num_iters: int | None = None
    max_model_len: int | None = None

    def argv(self) -> list[str]:
        args = [
            "--backend",
            "both",
            "--model",
            self.model,
            "--batch-size",
            str(self.batch_size),
            "--input-len",
            str(self.input_len),
            "--output-json",
            str(self.log.with_suffix(".json")),
        ]
        if self.max_model_len is not None:
            args.extend(["--max-model-len", str(self.max_model_len)])
        if self.num_iters is not None:
            args.extend(["--num-iters", str(self.num_iters)])
        return args


class Context:
    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.script = BENCHMARK_SCRIPT

    def run(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict] = []
        for scenario in self._suite_scenarios():
            row = self._run_scenario(scenario)
            if row is not None:
                summary_rows.append(row)
        summary_path = self.logs_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)
        logger.info("All runs complete. Logs in %s/", self.logs_dir)
        logger.info("Aggregate summary: %s", summary_path)

    def _suite_scenarios(self) -> list[Scenario]:
        scenarios: list[Scenario] = []

        model_slugs: dict[str, str] = {
            "Qwen/Qwen3-0.6B": "qwen3-0.6b",
            "Qwen/Qwen3-4B": "qwen3-4b",
            "Qwen/Qwen3-8B": "qwen3-8b",
        }
        batch_sizes = (1, 4, 8)
        input_lens = (512, 2048, 4096)

        for model, slug in model_slugs.items():
            max_model_len = MODEL_MAX_MODEL_LEN.get(model)
            for batch_size in batch_sizes:
                for input_len in input_lens:
                    tag = f"{slug}_bs{batch_size}_in{input_len}"
                    scenarios.append(
                        Scenario(
                            name=tag,
                            model=model,
                            batch_size=batch_size,
                            input_len=input_len,
                            log=self.logs_dir / f"cpu_decode_{tag}.log",
                            num_iters=15,
                            max_model_len=max_model_len,
                        )
                    )

        return scenarios

    def _run_scenario(self, scenario: Scenario) -> dict | None:
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
        json_path = scenario.log.with_suffix(".json")
        if not json_path.exists():
            return None
        with json_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        payload["scenario"] = scenario.name
        return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for per-run log/json files",
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
