"""Custom types."""
# Standard library
import os
import typing
from typing import Union

if typing.TYPE_CHECKING:
    # Standard library
    from os import PathLike  # noqa: F401  # imported but unused

# Index a numpy array
NDIndex_T = Union[int, slice, tuple[int, ...]]
NDIndices_T = Union[NDIndex_T, tuple[NDIndex_T, ...]]
# Note: While slice(None) is like :, None creates a new axis of unit length
OptNDIndex_T = Union[None, NDIndex_T]
OptNDIndices_T = Union[OptNDIndex_T, tuple[OptNDIndex_T, ...]]

# Represent a (system) path
PathLike_T = Union[str, os.PathLike]
PathLikeAny_T = Union[str, os.PathLike, bytes]
