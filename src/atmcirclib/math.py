"""Math utilities."""
from __future__ import annotations

# Third-party
import numpy as np


def step_ceil(f: float, step: float = 1.0, *, scale: bool = False) -> float:
    """Round a float to the next higher step.

    Arguments:
        f: Floating point number.

        step (optional): Size of steps to which ``f`` is rounded up.

        scale (optional): Scale ``step`` to order of magnitude of ``f``.

    Examples
        >>> step_ceil(0.5)
        1.0
        >>> step_ceil(0.3, 0.2)
        0.4
        >>> step_ceil(0.05, 0.1)
        0.1
        >>> step_ceil(7, 5.1)
        10.2
        >>> step_ceil(6, -4)
        8.0
        >>> step_ceil(-0.9, 0.2)
        -0.8

    """
    if scale:
        raise NotImplementedError("scaled step")
    step = abs(step)
    return float(np.ceil(f / step) * step)


if __name__ == "__main__":
    # Standard library
    import doctest

    doctest.testmod()
