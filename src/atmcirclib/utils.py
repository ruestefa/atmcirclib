"""Various utilities."""
from __future__ import annotations

# Standard library
import re
import sys
from functools import wraps
from typing import Any
from typing import Callable
from typing import Union

# Third-party
import xarray as xr


def partial_format(s: str, max_try: int = 99, /, **keys: Any) -> str:
    """Format ``s``, leaving format specifiers not in ``keys`` as they are."""
    # Note: s and max_try are positional-only to allow max_try to be omitted
    # when passing keys as `d: dict[str, str] = {...}; partial_format(s, **d)`
    # (otherwise mypy has to assume that max_try may be in the dict, so the dict
    # values must have a type compatible with int, e.g., Any; see also
    # https://github.com/python/mypy/issues/8038)
    for _ in range(max_try):
        try:
            return s.format(**keys)
        except KeyError as e:
            key = str(e).lstrip("\"'").rstrip("\"'")
            s = re.sub(r"(?<!{)(?P<key>{" + key + "(:[^}]*)?})(?!})", r"{\g<key>}", s)
    raise ValueError(f"partial formatting unsuccessful after {max_try} tries: '{s}'")


def exception_as_error(
    func: Callable[..., Any],
    print_: Callable[..., None] = print,
    exit_: Callable[[int], None] = sys.exit,
) -> Callable[..., Any]:
    """Decorate/wrap a function to replace an exception by an error message."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Catch any exception raised by ``func``, print the message and exit."""
        try:
            # mypy 0.941 thinks return value has type Any
            return func(*args, **kwargs)
        except Exception as ex:
            ex_name = type(ex).__name__
            msg = "error"
            if ex_name != "Exception":
                msg += f" ({ex_name})"
            msg += f": {ex}"
            print_(msg, file=sys.stderr)
            exit_(1)

    return wrapped


def rint(f: Union[float, xr.DataArray]) -> int:
    """Convert float into nearest int."""
    return int(round(float(f)))
