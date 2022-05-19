"""Plot a call graph created with ``PyCG``."""
from __future__ import annotations

# Standard library
import sys
from pathlib import Path

# Third-party
import click
import igraph
import pycg
import pycg.formats
import pycg.pycg

# First-party
from atmcirclib.click import CONTEXT_SETTINGS


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Create call graph(s) at ENTRY_POINT[S] with PyCG and plot them with iGraph",
)
@click.argument(
    "entry_points",
    metavar="ENTRY_POINT[S]",
    nargs=-1,
)
@click.option(
    "-o",
    "--out",
    help="Output file path",
    default="call_graph.svg",
)
def cli(entry_points: tuple[str, ...], out: str) -> None:
    """Command line interface."""
    call_graph = pycg.pycg.CallGraphGenerator(
        entry_points,
        package=None,
        max_iter=-1,
        operation="call-graph",
    )
    call_graph.analyze()
    call_graph_dict = pycg.formats.Simple(call_graph).generate()
    call_graph_tuples = [(k, v) for k, vs in call_graph_dict.items() for v in vs]
    graph = igraph.Graph.TupleList(call_graph_tuples)
    print(f"write {out}")
    suffix = Path(out).suffix.lstrip(".")
    if suffix == "svg":
        graph.write_svg(out)
    else:
        raise NotImplementedError(f"output file format '{suffix.upper()}' of {out}")


if __name__ == "__main__":
    sys.exit(cli())
