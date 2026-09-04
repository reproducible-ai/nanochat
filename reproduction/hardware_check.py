#!/usr/bin/env python3
"""Fail loudly unless this is the intended one-GPU Blackwell environment."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, found {torch.cuda.device_count()}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f"expected compute capability 12.0, found {capability}")
    distribution_version = importlib.metadata.version("torch")
    if "+" in distribution_version:
        raise RuntimeError(f"non-portable torch distribution version: {distribution_version}")
    left = torch.randn((256, 256), device="cuda")
    right = torch.randn((256, 256), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    if result.shape != (256, 256) or not torch.isfinite(result).all():
        raise RuntimeError("GPU matrix multiplication failed")
    evidence = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "gpu_count": torch.cuda.device_count(),
        "torch_distribution_version": distribution_version,
        "torch_runtime_cuda": torch.version.cuda,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/hardware.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
