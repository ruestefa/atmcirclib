"""Parse a Fortran namelist."""
from __future__ import annotations

# Standard library
import re
import sys
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Union


class NamelistParsingUnknownLineWarning(Warning):
    """Encountered unknown line during namelist parsing."""


class NamelistParsingUnknownVariableLineWarning(Warning):
    """Encountered unknown variable line during namelist parsing."""


def parse_namelist_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse a Fortran namelist file."""
    with open(path, "r") as f:
        return parse_namelist(f.read())


def parse_namelist(  # noqa: max-complexity: 25
    s: str, _dbg: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """Parse the content of a Fortran namelist file, passed as a string."""
    rxs_end = r"\s*(?:\!.*)?$"
    rxs_var = r"^\s*(?P<variable>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"
    rxs_start = r"^\s*"

    rx_blank = re.compile(rxs_start + rxs_end)
    rx_grp_start = re.compile(r"^\s*&(?P<group>[a-zA-Z_][a-zA-Z0-9_]*)" + rxs_end)
    rx_grp_end = re.compile(r"^\s*/\s*$" + rxs_end)

    rxs_val_num = r"-?\s*[0-9.]+"
    rxs_val_str = r"(?:'[^']*'" + r'|"[^"]*")'
    rxs_val_log = r"(?:\.[Tt][Rr][Uu][Ee]\.|\.[Ff][Aa][Ll][Ss][Ee]\.)"

    rx_var_num = re.compile(rxs_var + mult_vals(rxs_val_num) + rxs_end)
    rx_var_str = re.compile(rxs_var + mult_vals(rxs_val_str) + rxs_end)
    rx_var_log = re.compile(rxs_var + mult_vals(rxs_val_log) + rxs_end)
    rx_val_num = re.compile(rxs_start + mult_vals(rxs_val_num) + rxs_end)
    rx_val_str = re.compile(rxs_start + mult_vals(rxs_val_str) + rxs_end)
    rx_val_log = re.compile(rxs_start + mult_vals(rxs_val_log) + rxs_end)

    curr_group: str = ""
    curr_var: str = ""
    curr_vals: list[Any] = []

    grouped_vars: dict[str, list[dict[str, Any]]] = {}
    for line in s.split("\n"):
        if _dbg and curr_vals:
            print(f"{curr_group}::{curr_var} = {curr_vals}")

        # Blank line
        if m := rx_blank.match(line):
            if _dbg:
                print(f"BLANK       : {line}", file=sys.stderr)

        # Beginning of a group
        elif m := rx_grp_start.match(line):
            if _dbg:
                print(f"GROUP_START : {line}", file=sys.stderr)
            curr_group = m.group("group")
            if curr_group not in grouped_vars:
                grouped_vars[curr_group] = []
            grouped_vars[curr_group].append({})

        # End of a group
        elif m := rx_grp_end.match(line):
            if _dbg:
                print(f"GROUP_END   : {line}", file=sys.stderr)
            curr_group = ""
            curr_var = ""
            curr_vals.clear()

        # Number variable definition
        elif m := rx_var_num.match(line):
            if _dbg:
                print(f"VAR_NUM     : {line}", file=sys.stderr)
            curr_var = m.group("variable")
            curr_vals = extract_vals(m.group("value"), rxs_val_num, convert_num)
            assert curr_var not in grouped_vars[curr_group][-1]
            grouped_vars[curr_group][-1][curr_var] = curr_vals

        # Number variable line continuation
        elif m := rx_val_num.match(line):
            if _dbg:
                print(f"VAL_NUM     : {line}", file=sys.stderr)
            curr_vals += extract_vals(m.group("value"), rxs_val_num, convert_num)
            grouped_vars[curr_group][-1][curr_var] += curr_vals

        # String variable definition
        elif m := rx_var_str.match(line):
            if _dbg:
                print(f"VAR_STR     : {line}", file=sys.stderr)
            curr_var = m.group("variable")
            curr_vals = extract_vals(m.group("value"), rxs_val_str, convert_str)
            assert curr_var not in grouped_vars[curr_group]
            grouped_vars[curr_group][-1][curr_var] = curr_vals

        # String variable line continuation
        elif m := rx_val_str.match(line):
            if _dbg:
                print(f"VAL_STR     : {line}", file=sys.stderr)
            curr_vals += extract_vals(m.group("value"), rxs_val_str, convert_str)
            grouped_vars[curr_group][-1][curr_var] += curr_vals

        # Logical variable definition
        elif m := rx_var_log.match(line):
            if _dbg:
                print(f"VAR_LOG     : {line}", file=sys.stderr)
            curr_var = m.group("variable")
            curr_vals = extract_vals(m.group("value"), rxs_val_log, convert_log)
            assert curr_var not in grouped_vars[curr_group]
            grouped_vars[curr_group][-1][curr_var] = curr_vals

        # Logical variable line continuation
        elif m := rx_val_log.match(line):
            if _dbg:
                print(f"VAL_LOG     : {line}", file=sys.stderr)
            curr_vals += extract_vals(m.group("value"), rxs_val_log, convert_log)
            grouped_vars[curr_group][-1][curr_var] += curr_vals

        # Unknown variable (shouldn't happen)
        elif m := re.match(rxs_var, line):
            if _dbg:
                print(f"VAR_...     : {line}", file=sys.stderr)
            warnings.warn(f"'{line}'", NamelistParsingUnknownVariableLineWarning)

        # Unknown line (shouldn't happen)
        else:
            if _dbg:
                print(f"???         : {line}", file=sys.stderr)
            warnings.warn(f"'{line}'", NamelistParsingUnknownLineWarning)
    return grouped_vars


def mult_vals(rxs: str) -> str:
    """Extend a regex matching a single value to match multiple."""
    return r"(?P<value>(" + rxs + r"\s*,?\s*)+)"


def extract_vals(s: str, rxs: str, convert: Callable[[str], Any]) -> list[Any]:
    """Extract values from string using regex and convert them."""
    if raw := list(map(str.strip, re.findall(rxs, s))):
        return [convert(v) for v in raw]
    return []


def convert_num(s: str) -> Union[int, float]:
    """Convert a number string to an int or float."""
    if (i := int(float(s))) == (f := float(s)):
        return i
    return f


def convert_str(s: str) -> str:
    """Remove leading and trailing quotes from a string."""
    return s.strip("'\"")


def convert_log(s: str) -> bool:
    """Convert a Fortran logical string to a bool."""
    return {".true.": True, ".false.": False}[s.lower()]


def check_unique_var_names_across_groups(  # noqa: max-complexity: 11
    grouped_vars: dict[str, list[dict[str, Any]]]
) -> None:
    """Check that all variable names are unique across namelist groups."""
    # Flatten groups
    grouped_var_names: dict[str, set[str]] = {g: set() for g in set(grouped_vars)}
    group_vars: list[dict[str, Any]]
    subroup_vars: dict[str, Any]
    for group, group_vars in grouped_vars.items():
        for subgroup_vars in group_vars:
            for var_name in subgroup_vars:
                grouped_var_names[group].add(var_name)
    var_names = [v for vs in grouped_var_names.values() for v in vs]
    if len(var_names) > len(set(var_names)):
        dupes: dict[str, list[str]] = {}
        for group, var_names_ in grouped_var_names.items():
            for var_name in var_names_:
                for inner_group, inner_var_names in grouped_var_names.items():
                    if inner_group != group:
                        if var_name in inner_var_names:
                            if var_name not in dupes:
                                dupes[var_name] = [group]
                            dupes[var_name].append(inner_group)
        dupes_s = "; ".join(
            [f"{var_name} in {', '.join(groups)}" for var_name, groups in dupes.items()]
        )
        raise Exception(f"var names not unique: {dupes_s}")


def flatten_groups(
    grouped_vars: dict[str, list[dict[str, list[Any]]]],
    check_unique_vars: bool = True,
    merge_dupe_groups: bool = True,
) -> dict[str, list[Any]]:
    """Flatten namelist groups.

    Args:
        grouped_vars: Variables and their values by group name, whereby there
            may be multiple instances of a group.

        check_unique_vars (optional): Check that variables names are unique to
            their group.

        merge_dupe_groups (optional): Merge duplicate groups by collecting all
            variables and by merging the values of variables occurring in
            multiple instances of a group; otherwise, only the last instance
            of each group is selected.

    """
    if check_unique_vars:
        check_unique_var_names_across_groups(grouped_vars)
    vars: dict[str, list[Any]] = {}
    group_vars: list[dict[str, Any]]
    subroup_vars: dict[str, Any]
    if not merge_dupe_groups:
        for group, group_vars in grouped_vars.items():
            grouped_vars[group] = [group_vars[-1]]
    for group, group_vars in grouped_vars.items():
        for subgroup_vars in group_vars:
            for var_name, values in subgroup_vars.items():
                if var_name not in vars:
                    vars[var_name] = []
                vars[var_name].extend(values)
    return vars
