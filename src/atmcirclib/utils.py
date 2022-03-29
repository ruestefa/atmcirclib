"""Various utilities."""
from __future__ import annotations

# Standard library
import re
from typing import Any


def partial_format(s: str, *, max_try: int = 99, **keys: Any) -> str:
    """Format ``s``, leaving format specifiers not in ``keys`` as they are."""
    for i in range(max_try):
        try:
            return s.format(**keys)
        except KeyError as e:
            key = str(e).lstrip("\"'").rstrip("\"'")
            s = re.sub(r"(?<!{)(?P<key>{" + key + "(:[^}]*)?})(?!})", r"{\g<key>}", s)
    raise ValueError(f"partial formatting unsuccessful after {max_try} tries: '{s}'")
