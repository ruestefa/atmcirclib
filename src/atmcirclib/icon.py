"""Work with ICON setup and output files."""
from __future__ import annotations

# Standard library
import datetime as dt
import re
from pathlib import Path
from typing import Union

# Third-party
import xarray as xr

# First-party
from atmcirclib.fortran.namelist_parser import flatten_groups
from atmcirclib.fortran.namelist_parser import parse_namelist_file
from atmcirclib.utils import partial_format
from atmcirclib.utils import rint

PathLike_T = Union[Path, str]


def find_namelist_path(
    dir: PathLike_T,
    master: str = "icon_master.namelist",
    param: str = "model_namelist_filename",
) -> Path:
    """Obtain path to full namelist of ICON run by parsing master namelist.

    Arguments:
        dir: Path to ICON run directory.

        master (optional): File name of master namelist.

        param (optional): Name of master namelist parameter containing
            the name of the file name of the full namelist.

    """
    dir = Path(dir)
    master_path = dir / master
    rx = re.compile(r"^\s*" + param + r'\s*=\s*"(?P<name>[^"]+)"')
    with open(master_path, "r") as f:
        for line in f.readlines():
            if m := rx.match(line):
                namelist_name = m.group("name")
                break
        else:
            raise Exception(
                "param not found in namelist file with regex"
                f"\n{param=}\n{master_path=}\n{rx.pattern=}"
            )
    return dir / namelist_name


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


def convert_icon_time_step(
    time: xr.DataArray,
) -> dt.datetime:
    """Convert a time step from ICON format to a datetime object."""
    assert time.shape == (), f"unexpected time shape: {time.shape}"
    assert "%Y%m%d.%f" in time.units, f"unexpected time units: {time.units}"
    year = int(time / 1e4)
    month = int(((int(time) / 1e4) % 1) * 1e2)
    day = int(((int(time) / 1e2) % 1) * 1e2)
    tot_secs = rint((float(time) % 1) * 24 * 3600)
    hour = int(tot_secs / 3600)
    min_ = int((tot_secs % 3600) / 60)
    sec = tot_secs % 3600 % 60
    return dt.datetime(year, month, day, hour, min_, sec)


def convert_icon_time(time: xr.DataArray) -> xr.DataArray:
    """Convert time steps from ICON format to datetime objects."""
    return xr.DataArray(
        data=list(map(convert_icon_time_step, time)), coords={"time": time}
    )
