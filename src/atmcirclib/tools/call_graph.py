"""Plot a call graph created with ``PyCG``."""
from __future__ import annotations

# Standard library
import os
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Optional

# Third-party
import click
import pycg
import pycg.formats
from pycg.pycg import CallGraph
from pycg.pycg import CallGraphGenerator
from pygraphviz import AGraph

# First-party
from atmcirclib.click import CONTEXT_SETTINGS


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Create call graph(s) at ENTRY_POINT[S] with PyCG and plot them with iGraph",
)
@click.option(
    "-e",
    "--entry-point",
    "entry_points",
    help="Entry point to be processed; may be repeated",
    type=click.Path(exists=True),
    multiple=True,
)
@click.option(
    "-p",
    "--package",
    help="Package to be analyzed",
    default=None,
)
@click.option(
    "-i",
    "--include",
    "includes",
    metavar="KEY",
    help="Only include elements that start with KEY; may be repeated",
    default=(),
    multiple=True,
)
@click.option(
    "-x",
    "--exclude",
    "excludes",
    metavar="KEY",
    help="Exclude elements that start with KEY; may be repeated",
    default=(),
    multiple=True,
)
@click.option(
    "--exclude-builtin/--no-exclude-builtin",
    help="Exclude builtins",
    default=True,
)
@click.option(
    "--shorten/--no-shorten",
    help="Shorten names by removing common prefixes",
    default=True,
)
@click.option(
    "-l",
    "--layout",
    help=(
        "Graph layout; options: dot, neato, twopi, circo, fdp, osage, patchwork, sfdp"
        " (see https://graphviz.org/docs/layouts)"
    ),
    default="fdp",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    help="Output file path",
    default="call_graph.svg",
)
def cli(
    entry_points: tuple[str, ...],
    package: Optional[str],
    includes: tuple[str, ...],
    excludes: tuple[str, ...],
    exclude_builtin: bool,
    shorten: bool,
    layout: str,
    out_path: str,
) -> None:
    """Command line interface."""
    if exclude_builtin:
        excludes = tuple(set(excludes) | {"<builtin>."})
    cg: CallGraph = CallGraphGenerator(
        entry_points, package=package, max_iter=-1, operation="call-graph"
    )
    cg.analyze()
    cgd = (
        CallGraphDict.from_call_graph(cg)
        .retain(includes)
        .remove(excludes)
        .shorten(shorten)
    )
    ag = AGraph(cgd, directed=True, overlap=False)
    ag.draw(out_path, prog=layout)


class CallGraphDict(dict[str, list[str]]):
    """A dictionary representation of a call graph."""

    def retain(self, prefixes: Optional[Collection[str]]) -> CallGraphDict:
        """Retain only names that start with any of the given strings."""
        obj = self.copy()
        if not prefixes:
            return obj

        def is_included(name: str) -> bool:
            """Check whether a name is included."""
            return any(name.startswith(incl) for incl in (prefixes or []))

        cgd = {
            k: [v for v in vs if is_included(v)]
            for k, vs in obj.items()
            if is_included(k)
        }
        return type(self)(cgd)

    def remove(self, prefixes: Optional[Collection[str]]) -> CallGraphDict:
        """Remove names that start with any of the given strings."""
        obj = self.copy()
        if not prefixes:
            return obj
        for excl in prefixes:
            cdg = {
                k: [v for v in vs if not v.startswith(excl)]
                for k, vs in obj.items()
                if not k.startswith(excl)
            }
            obj = type(self)(cdg)
        return obj

    def shorten(self, value: bool = True) -> CallGraphDict:
        """Remove common prefix (separated by period) from all names."""
        obj = self.copy()
        if not value:
            return obj
        names = list(set(obj.keys()) | {v for vs in obj.values() for v in vs})
        prefix = os.path.commonprefix(names)
        if all(name.startswith(prefix + ".") or name == prefix for name in names):
            prefix += "."
        if "." not in prefix:
            return obj
        prefix = prefix.removesuffix(Path(prefix).suffix[1:])[:-1]
        if not prefix:
            return obj

        def rm_pre(name: str) -> str:
            """Remove prefix from name."""
            if name == prefix:
                return "."
            return name[len(prefix) :]

        cdg = {rm_pre(k): [rm_pre(v) for v in vs] for k, vs in obj.items()}
        return type(self)(cdg)

    def copy(self) -> CallGraphDict:
        """Return a copy of the instance."""
        return type(self)(dict(self))

    @classmethod
    def from_call_graph(cls, cg: CallGraph) -> CallGraphDict:
        """Create a call graph dict from a PyCG call graph object."""
        return cls(pycg.formats.Simple(cg).generate())


if __name__ == "__main__":
    sys.exit(cli())
