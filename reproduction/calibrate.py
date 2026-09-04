#!/usr/bin/env python3
"""Measure depth-16 steady-state time and fail if the full pretrain is too long."""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("outputs/nanochat")
LOG_PATH = Path("outputs/calibration-depth16.log")
FULL_ITERATIONS = 3584
MAX_PROJECTED_PRETRAIN_SECONDS = 4 * 60 * 60


def project_pretrain_seconds(step_ms: list[float]) -> float:
    if len(step_ms) < 4:
        raise RuntimeError("calibration did not report enough timed steps")
    steady_ms = statistics.median(step_ms[-3:])
    return steady_ms / 1000 * FULL_ITERATIONS


def enforce_projection_ceiling(projected_seconds: float) -> None:
    if projected_seconds > MAX_PROJECTED_PRETRAIN_SECONDS:
        raise RuntimeError("projected pretraining time exceeds the run ceiling")


def main() -> int:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(BASE_DIR)
    command = [
        sys.executable,
        "-m",
        "scripts.base_train",
        "--depth=16",
        "--target-param-data-ratio=8",
        "--device-batch-size=32",
        "--fp8",
        "--num-iterations=5",
        "--model-tag=calibration-depth16",
        "--core-metric-every=-1",
        "--eval-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--run=dummy",
    ]
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        print(LOG_PATH.read_text(encoding="utf-8", errors="replace"))
        return completed.returncode

    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    step_ms = [float(value) for value in re.findall(r"dt: ([0-9.]+)ms", log_text)]
    if len(step_ms) < 4:
        print(log_text)
    projected_seconds = project_pretrain_seconds(step_ms)
    steady_ms = statistics.median(step_ms[-3:])
    checkpoint_dir = BASE_DIR / "base_checkpoints" / "calibration-depth16"
    if not list(checkpoint_dir.glob("model_*.pt")):
        raise RuntimeError("calibration produced no model checkpoint")

    result = {
        "depth": 16,
        "full_iterations": FULL_ITERATIONS,
        "measured_step_ms": step_ms,
        "steady_step_ms": steady_ms,
        "projected_pretrain_seconds": projected_seconds,
        "maximum_projected_pretrain_seconds": MAX_PROJECTED_PRETRAIN_SECONDS,
    }
    Path("outputs/calibration-depth16.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    enforce_projection_ceiling(projected_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
