"""Work with Fortran files."""
from __future__ import annotations

# Local
from .namelist_parser import parse_namelist
from .namelist_parser import parse_namelist_file

__all__: list[str] = [
    "parse_namelist",
    "parse_namelist_file",
]
