"""Work with ICON setup and output files."""
from __future__ import annotations

# Standard library
from pathlib import Path

# First-party
from atmcirclib.fortran.namelist_parser import flatten_groups
from atmcirclib.fortran.namelist_parser import parse_namelist_file
from atmcirclib.utils import partial_format


def format_icon_params(
    *,
    namelist_path: Path,
    title_tmpl: str,
    partial: bool = True,
) -> str:
    """Format template string by inserting ICON namelist parameters.

    Variables can be included with Python-style format keys: ``{<var>[:...]}``.

    The following regular variables are supported:
        'abs_step': Absolute time step.

        'rel_step': Time step relative to the start of the simulation.

    In addition, any ICON namelist parameters can be included by prefacing their
    name with 'nl__', e.g., 'nl__dtime' to include the model time step 'dtime'.

    If a parameter contains multiple values (e.g., output variables or in case
    of nested domains), or if a parameter is passed multiple times in successive
    instances of a namelist group (e.g, multiple output groups), the values are
    concatenated in order of appearance. (Support for such multi-value
    parameters is rudimentary at this point.)

    If ``partial`` is true, additional, non-ICON format parameters are ignored.
    Set it to false to ensure that no such keys are present.

    """
    grouped_vars = parse_namelist_file(namelist_path)
    # Collect all variables across groups, assuming (and checking) that all
    # variable names are unique across groups (if they weren't, the group
    # should also be specified in the format keys, e.g., ``nl__<group>__<var>``,
    # but AFAIK this is not necessary in ICON)
    vars = flatten_groups(grouped_vars, check_unique_vars=True)
    vars_fmtd: dict[str, str] = {}
    for name, vals in vars.items():
        vals_fmtd = [f"'{v}'" if isinstance(v, str) else str(v) for v in vals]
        vars_fmtd[f"nl__{name}"] = ",".join(vals_fmtd)
    if partial:
        return partial_format(title_tmpl, **vars_fmtd)
    return title_tmpl.format(**vars_fmtd)
