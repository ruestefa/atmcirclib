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

        param (optional): Name of master namelist parameter containing the name
            of the file name of the full namelist.

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
    s: str,
    /,
    path: PathLike_T,
    *,
    partial: bool = True,
) -> str:
    """Format template string by inserting ICON namelist parameters.

    Arguments:
        s: Format string with '{...}'-style keys; insert ICON namelist parameters
            with '{nl__<param>[:...]}', e.g., ``{nl__dtime:02d}``; may contain
            other format keys if ``partial`` is true, in which case those are
            ignored; for details on multi-value parameters and parameter groups,
            see below.

        path: Path to an ICON namelist file or an ICON run directory; if the
            latter, the path to the full namelist file is obtained with
            ``find_namelist_path``.

        partial (optional): Ignore format keys of non-ICON parameters; if false,
            such keys will trigger an exception.

    If a parameter contains multiple values (e.g., output variables or in case
    of nested domains), or if a parameter is passed multiple times in successive
    instances of a namelist group (e.g, multiple output groups), the values are
    concatenated in order of appearance. (Support for such multi-value
    parameters is rudimentary at this point.)

    """
    path = Path(path)
    if path.is_file():
        namelist_path = path
    elif path.is_dir():
        namelist_path = find_namelist_path(path)
    else:
        raise ValueError(
            f"path neither points to a namelist file nor a run directory: {path}"
        )
    grouped_vars = parse_namelist_file(namelist_path)
    # Collect all variables across groups, assuming (and checking) that all
    # variable names are unique across groups (if they weren't, the group
    # should also be specified in the format keys, e.g., ``nl__<group>__<var>``,
    # but AFAIK this is not necessary in ICON)
    vars = flatten_groups(grouped_vars, check_unique_vars=True)
    vars_fmtd: dict[str, str] = {}
    for name, vals in vars.items():
        key = f"nl__{name}"
        # Note: Formatting of values is very rudimentary, especially for
        # multi-value parameters where individual values cannot be formatted
        # (e.g., no. decimal digits or so); this should be refined as need
        # arises (but it's good enough for simple use cases)
        if len(vals) == 1 and not isinstance(vals[0], str):
            # Leave single values as they are to enable formatting options like
            # number formats, decimal digits etc.
            vars_fmtd[key] = vals[0]
        else:
            # Add quotes to strings; format multi-values with plain ``str()``
            vars_fmtd[key] = ",".join(
                [f"'{val}'" if isinstance(val, str) else str(val) for val in vals]
            )
    # Relating to the note above, note that there are no checks for cases where
    # formatting options (like decimal digits) are specified for multi-value
    # parameters, for which this is not supported; in that case, ``str.format``
    # will just fail
    if partial:
        return partial_format(s, **vars_fmtd)
    return s.format(**vars_fmtd)


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
