"""Plot a call graph created with ``PyCG``."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path
from typing import Any
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


def create_call_graph(
    entry_points: tuple[str, ...], package: Optional[str]
) -> CallGraph:
    """Create a call graph with PyCG."""
    cg = CallGraphGenerator(
        entry_points,
        package=package,
        max_iter=-1,
        operation="call-graph",
    )
    cg.analyze()
    return cg


def prepare_plot_graph(cg: CallGraph, layout: str) -> AGraph:
    """Convert PyCG call graph into iGraph graph for plotting."""
    cg = pycg.formats.Simple(cg).generate()
    ag = AGraph(cg, directed=True, overlap=False)
    ag.layout(layout)
    return ag


def write_graph(ag: AGraph, path: str) -> None:
    """Write graph to disk."""
    print(f"write {path}")
    suffix = Path(path).suffix.lstrip(".")
    if suffix == "svg":
        ag.draw(path)
    else:
        raise NotImplementedError(f"output file format '{suffix.upper()}' of {path}")


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
    "-o",
    "--out",
    "out_path",
    help="Output file path",
    default="call_graph.svg",
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
def cli(layout: str, out_path: str, **kwargs: Any) -> None:
    """Command line interface."""
    cg = create_call_graph(**kwargs)
    ag = prepare_plot_graph(cg, layout)
    write_graph(ag, out_path)


if __name__ == "__main__":
    sys.exit(cli())
