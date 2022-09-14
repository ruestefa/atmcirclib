"""Subpackage ``atmcirclib.deriv``."""
from __future__ import annotations

# Standard library
from warnings import warn

try:
    # Local
    from . import ext
except ImportError as e:
    warn(f"cannot import atmcirclib.deriv.ext: {e}")
    ext = None  # type: ignore
