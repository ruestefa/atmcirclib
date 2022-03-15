"""Set up the project."""
from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

# Third-party
from pkg_resources import parse_requirements
from setuptools import find_packages
from setuptools import setup

PackageDataT = Dict[str, List[str]]


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
    "name": "atmcirclib",
    "version": "0.0.1",
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

python: str = ">= 3.9"

# Runtime dependencies: top-level and unpinned (only critical version restrictions)
with open("requirements.in") as f:
    requirements: list[str] = list(map(str, parse_requirements(f.readlines())))

# Format: command=package.module:function
scripts: list[str] = []

package_data: PackageDataT = find_py_typed()

setup(
    python_requires=python,
    install_requires=requirements,
    entry_points={"console_scripts": scripts},
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data=package_data,
    include_package_data=True,
    zip_save=False,
    **metadata,
)
