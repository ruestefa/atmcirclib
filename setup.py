"""Set up the project."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path

# Third-party
from skbuild import setup

# Obtain version and root of currently active Python environment for cmake
curr_python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
curr_python_root: str = str(Path(sys.executable).parent.parent)  # remove `bin/python`

# Arguments passed to cmake by scikit-build
cmake_args: list[str] = [
    f"-DCMAKE_PREFIX_PATH={curr_python_root}",
    f"-DCMAKE_PYTHON_VERSION={curr_python_version}",
]

setup(
    cmake_args=cmake_args,
)
