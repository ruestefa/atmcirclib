"""Plot a call graph created with ``PyCG``."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path
from typing import Any
from typing import Optional

# Third-party
import click
import igraph
import pycg
import pycg.formats
import pycg.pycg

# First-party
from atmcirclib.click import CONTEXT_SETTINGS


def create_call_graph(
    entry_points: tuple[str, ...], package: Optional[str]
) -> pycg.pycg.CallGraph:
    """Create a call graph with PyCG."""
    call_graph = pycg.pycg.CallGraphGenerator(
        entry_points,
        package=package,
        max_iter=-1,
        operation="call-graph",
    )
    call_graph.analyze()
    return call_graph


def prepare_plot_graph(call_graph: pycg.pycg.CallGraph) -> igraph.Graph:
    """Convert PyCG call graph into iGraph graph for plotting."""
    call_graph_dict = pycg.formats.Simple(call_graph).generate()
    call_graph_tuples = [(k, v) for k, vs in call_graph_dict.items() for v in vs]
    return igraph.Graph.TupleList(call_graph_tuples)


def write_graph(graph: igraph.Graph, path: str) -> None:
    """Write graph to disk."""
    print(f"write {path}")
    suffix = Path(path).suffix.lstrip(".")
    if suffix == "svg":
        graph.write_svg(path)
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
def cli(out_path: str, **kwargs: Any) -> None:
    """Command line interface."""
    call_graph = create_call_graph(**kwargs)
    plot_graph = prepare_plot_graph(call_graph)
    write_graph(plot_graph, out_path)


if __name__ == "__main__":
    sys.exit(cli())
