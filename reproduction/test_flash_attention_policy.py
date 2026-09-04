#!/usr/bin/env python3
"""Check that precompiled attention kernels are selected only where supported."""

from nanochat.flash_attention import _kernel_repo_for_compute_major


def main() -> None:
    assert _kernel_repo_for_compute_major(8) == "kernels-community/flash-attn3"
    assert _kernel_repo_for_compute_major(9) == "varunneal/flash-attention-3"
    assert _kernel_repo_for_compute_major(12) is None
    print("flash-attention architecture policy: PASS")


if __name__ == "__main__":
    main()
