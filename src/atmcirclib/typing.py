"""Custom types."""
# Standard library
import os
import typing
from typing import Union

if typing.TYPE_CHECKING:
    # Standard library
    from os import PathLike  # noqa: F401  # imported but unused

PathLike_T = Union[str, os.PathLike]
PathLikeAny_T = Union[str, os.PathLike, bytes]
