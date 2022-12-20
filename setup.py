"""Set up the project."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path

# Third-party
from setuptools import find_packages
from skbuild import setup

PROJECT_NAME: str = "atmcirclib"

# Obtain version and root of currently active Python environment for cmake
curr_python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
curr_python_root: str = str(Path(sys.executable).parent.parent)  # remove `bin/python`

# Arguments passed to cmake by scikit-build
cmake_args: list[str] = [
    f"-DCMAKE_PREFIX_PATH={curr_python_root}",
    f"-DCMAKE_PYTHON_VERSION={curr_python_version}",
]

if sys.argv[1] == "develop":
    # `packages=find_packages("src")` is broken for projects with subpackages,
    # so only list top-level package(s) during development install
    # src: https://github.com/scikit-build/scikit-build/issues/546
    packages = [PROJECT_NAME]
else:
    packages = find_packages("src")

setup(
    packages=packages,
    package_dir={"": "src"},
    cmake_args=cmake_args,
)
