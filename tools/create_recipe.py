#!/usr/bin/env python
"""Create conda recipe files ``meta.yaml`` etc. based on ``setup.py`` etc.

Becomces obsolete once ``grayskull`` can handle gitlab repos (v1.1.2 cannot).

"""
from __future__ import annotations

# Standard library
import dataclasses as dc
import importlib
import os
import pprint as pp
import sys
import tempfile
import typing
from textwrap import dedent
from types import ModuleType
from typing import Any
from typing import Optional
from typing import Union

if typing.TYPE_CHECKING:
    from os import PathLike  # noqa

# Third-party
import click

# Types
PathLike_T = Union[str, bytes, os.PathLike]


@dc.dataclass
class SetupPyFile:
    """File ``setup.py``."""

    path: PathLike_T = "setup.py"

    def __post_init__(self) -> None:
        """Finish initialization."""
        self._mod: ModuleType = self._import_partial()

    def get_metadata(self) -> dict[str, Any]:
        """Get module attribute ``metadata``."""
        return dict(getattr(self._mod, "metadata"))

    def get_requirements(self) -> list[str]:
        """Get module attribute ``requirements``."""
        return list(getattr(self._mod, "requirements"))

    def _import_partial(self) -> ModuleType:
        """Import partial ``setup.py`` file (w/o ``setup(...)``) as module."""
        with open(self.path) as f:
            # Read setup.py up to ``setup(...)``
            lines: list[str] = []
            for line in f.readlines():
                if line.startswith("setup("):
                    break
                lines.append(line)
        with tempfile.NamedTemporaryFile("a") as f:
            # Import abbreviated setup.py
            f.writelines(lines)
            f.seek(0)  # reset pointer to beginning
            return importlib.machinery.SourceFileLoader("setup", f.name).load_module()


@dc.dataclass
class Metadata:
    """Package metadata."""

    name: str
    version: str
    author: str
    author_email: str
    url: str
    keywords: str
    description: str
    license: str = "MIT"
    long_description: str = ""
    classifiers: list[str] = dc.field(default_factory=list)

    def __post_init__(self) -> None:
        """Finish initialization."""
        if self.url.startswith("https://"):
            self.url = self.url.replace("https://", "git+ssh://git@")


class TestMetadata(Metadata):
    """Package metadata with hardcoded values for testing."""

    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__(
            name="testtool",
            version="1.2.3",
            author="Foobert Barson",
            author_email="foo.bar@baz.com",
            description="Test tool",
            url="git@ssh://git@github.com/foobar/baz.git",
            keywords="foo bar baz",
        )


@dc.dataclass
class MetaYamlFile:
    """Main conda recipe file ``conda.yaml``."""

    metadata: Metadata
    requirements: list[str]
    indent: int = 2

    def format(self) -> str:
        """Return file content as string."""
        s = """\
        {{% set name = "{name}" %}}
        {{% set version = "{version}" %}}
        {{% set build = "0" %}}

        package:
        {ind}name: {{{{ name | lower }}}}
        {ind}version: {{{{ version }}}}

        source:
        {ind}git_rev: v{{{{ version }}}}
        {ind}git_url: {url}

        build:
        {ind}noarch: python
        {ind}number: {{{{ build }}}}
        {ind}script: python -m pip install --no-deps --ignore-installed .

        requirements:
        {ind}host:
        {ind}{ind}- python
        {ind}{ind}- pip
        {ind}run:
        {requirements_fmtd}

        test:
        {ind}imports:
        {ind}{ind}- {{{{ name }}}}

        about:
        {ind}license: {license}
        {ind}summary: {description}
        {ind}description: |
        {long_description_fmtd}
        """
        return dedent(s).format(
            ind=" " * self.indent,
            requirements_fmtd=self._format_requirements(),
            long_description_fmtd=self._indent(self.metadata.long_description, 2),
            **dc.asdict(self.metadata),
        )

    def write(self, path: Optional[PathLike_T] = None, *, _indent: int = 0) -> None:
        """Write file to disk."""
        content = self.format()
        if _indent:
            content = " " * _indent + content.replace(
                "\n", f"\n{' ' * _indent}"
            ).rstrip(" ")
        if path is None or path == "-":
            sys.stdout.write(content)
        else:
            with open(path, "w") as f:
                f.write(content)

    def _format_requirements(self) -> str:
        """Format requirements."""
        return self._indent("- " + "\n- ".join(self.requirements), 2)

    def _indent(self, s: str, n: int = 1) -> str:
        """Indent all lines of ``s`` by ``n * self.indent``."""
        ind = " " * self.indent * n
        return f"{ind}" + f"\n{ind}".join(s.split("\n"))


@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "-o",
    "--outfile",
    help="Outfile file path; pass - to write to standard output.",
    default="recipe/meta.yaml",
)
def cli(ctx: click.Context, **kwargs: Any) -> None:
    """Command line interface."""
    if ctx.invoked_subcommand is None:
        main(**kwargs)


def main(outfile: str) -> None:
    """Main program."""
    setup_py = SetupPyFile()
    yaml = MetaYamlFile(
        metadata=Metadata(**setup_py.get_metadata()),
        requirements=setup_py.get_requirements(),
    )
    yaml.write(outfile)


@cli.command("test")
def test() -> None:
    """Run tests."""
    print("## Test 1: Format meta.yaml based on hardcoded metadata")
    indent = 0
    print(f"\n{' ' * indent}```yaml: meta.yaml")
    yaml = MetaYamlFile(TestMetadata(), requirements=["hello_world>=15.4"])
    yaml.write(_indent=indent)
    print(f"{' ' * indent}```\n")

    print("## Test 2: Obtain metadata from setup.py")
    metadata = SetupPyFile().get_metadata()
    print(f"\n```python: metadata\n{pp.pformat(metadata)}\n```\n")

    print("## Test 3: Obtain requirements from setup.py")
    requirements = SetupPyFile().get_requirements()
    print(f"\n```python: requirements\n{pp.pformat(requirements)}\n```\n")


if __name__ == "__main__":
    sys.exit(cli())
