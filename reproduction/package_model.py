#!/usr/bin/env python3
"""Package the final depth-16 checkpoint and tokenizer as one artifact."""

from __future__ import annotations

import tarfile
from pathlib import Path


BASE_DIR = Path("outputs/nanochat")
ARCHIVE = Path("outputs/nanochat-depth16.tar.gz")


def main() -> None:
    checkpoint_dir = BASE_DIR / "chatsft_checkpoints" / "depth16"
    model_files = sorted(checkpoint_dir.glob("model_*.pt"))
    meta_files = sorted(checkpoint_dir.glob("meta_*.json"))
    tokenizer_files = sorted((BASE_DIR / "tokenizer").glob("*"))
    if len(model_files) != 1 or len(meta_files) != 1 or not tokenizer_files:
        raise RuntimeError("expected one final model, one metadata file, and tokenizer files")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in [*model_files, *meta_files, *tokenizer_files]:
            archive.add(path, arcname=path.relative_to(BASE_DIR))
    if ARCHIVE.stat().st_size == 0:
        raise RuntimeError("model archive is empty")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
