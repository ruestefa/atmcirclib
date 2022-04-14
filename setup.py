"""Set up the project."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

# Third-party
from pkg_resources import parse_requirements
from setuptools import find_packages
from skbuild import setup

PackageDataT = Dict[str, List[str]]

PROJECT_NAME: str = "atmcirclib"
PROJECT_VERSION: str = "0.2.1"
PYTHON_REQUIRES: str = ">=3.9"


def read_present_files(paths: Sequence[str]) -> str:
    """Read the content of files in ``paths`` that exist, ignoring others."""
    contents: list[str] = []
    for path in paths:
        try:
            with open(path, "r") as f:
                contents += ["\n".join(map(str.strip, f.readlines()))]
        except FileNotFoundError:
            continue
    return "\n\n".join(contents)


def find_py_typed(
    package_data: Optional[PackageDataT] = None, src: str = "src"
) -> PackageDataT:
    """Find all packages in ``src`` that contain a ``py.typed`` file.

    The returned dictionary can used as (or inserted in) ``package_data``.

    """
    if package_data is None:
        package_data = {}
    for path in Path(src).glob("*/py.typed"):
        package_name = path.parent.name
        if package_name not in package_data:
            package_data[package_name] = []
        package_data[package_name].append(str(path))
    return package_data


description_files: list[str] = [
    "README",
    "README.md",
    "README.rst",
    "HISTORY",
    "HISTORY.md",
    "HISTORY.rst",
]

metadata: dict[str, Any] = {
    "name": PROJECT_NAME,
    "version": PROJECT_VERSION,
    "description": "Python library of Atmospheric Circulation group",
    "long_description": read_present_files(description_files),
    "author": "Stefan Ruedisuehli",
    "author_email": "stefan.ruedisuehli@env.ethz.ch",
    "url": "https://git.iac.ethz.ch/atmcirc/tools/atmcirclib.git",
    "keywords": "atmospheric circulation",
}

classifiers: list[str] = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
]
metadata["classifiers"] = classifiers

# Runtime dependencies (top-level and unpinned)
with open("requirements.in") as f:
    install_requires: list[str] = list(map(str, parse_requirements(f.readlines())))

# Format: command=package.module:function
scripts: list[str] = []

# Obtain version and root of currently active Python environment for cmake
curr_python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
curr_python_root: str = str(Path(sys.executable).parent.parent)  # remove `bin/python`

# Arguments passed to cmake by scikit-build
cmake_args: list[str] = [
    f"-DCMAKE_PREFIX_PATH={curr_python_root}",
    f"-DCMAKE_PYTHON_VERSION={curr_python_version}",
]

setup(
    # `packages=find_packages("src")` is broken for projects with subpackages,
    # so only list top-level package(s) during development install
    # src: https://github.com/scikit-build/scikit-build/issues/546
    # packages=find_packages("src"),
    packages=[PROJECT_NAME] if sys.argv[1] == "develop" else find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": scripts},
    package_data=find_py_typed(),
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=install_requires,
    cmake_args=cmake_args,
    **metadata,
)
