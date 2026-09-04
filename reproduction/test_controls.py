#!/usr/bin/env python3
"""Positive and negative controls for the calibration ceiling."""

from calibrate import enforce_projection_ceiling, project_pretrain_seconds


def main() -> None:
    fast_projection = project_pretrain_seconds([9000, 3000, 3000, 3000])
    enforce_projection_ceiling(fast_projection)

    slow_projection = project_pretrain_seconds([9000, 5100, 5100, 5100])
    try:
        enforce_projection_ceiling(slow_projection)
    except RuntimeError:
        pass
    else:
        raise AssertionError("negative control did not reject a slow projection")

    try:
        project_pretrain_seconds([3000, 3000, 3000])
    except RuntimeError:
        pass
    else:
        raise AssertionError("negative control accepted too few samples")

    print("calibration controls: PASS")


if __name__ == "__main__":
    main()
