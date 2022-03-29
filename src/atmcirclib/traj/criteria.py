"""Work with trajectories."""
from __future__ import annotations

# Standard library
import abc
import dataclasses as dc
from collections import UserList
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

# Third-party
import numpy as np
import numpy.typing as npt

# First-party
from atmcirclib.cosmo import COSMOGridDataset

if TYPE_CHECKING:
    # Local
    from .traj_dataset import TrajDataset

__all__: list[str] = [
    "AllCriterion",
    "BoundaryZoneCriterion",
    "Criteria",
    "CriteriaFormatter",
    "Criterion",
    "LeaveDomainCriterion",
    "VariableCriterion",
    "VariableCriterionFormatter",
]


def sfmt(v: Any, q: str = "'") -> str:
    """If ``v`` is a string, adding quotes, otherwise convert to string."""
    if isinstance(v, str):
        return f"{q}{v}{q}"
    return str(v)


class Criterion(abc.ABC):
    """Base class of criteria to selection trajectories."""

    @abc.abstractmethod
    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        ...

    @abc.abstractmethod
    def invert(self) -> Criterion:
        """Invert the criterion."""
        ...

    def dict(self) -> dict[str, Any]:
        """Return dictionary reprentation with all instantiation arguments."""
        # pylint: disable=R0201  # no-self-use
        return {}

    def format(self, mode: str = "human") -> str:
        """Format criterion to a string."""
        methods_by_mode = {
            "human": self._format_human,
            "file": self._format_file,
        }
        try:
            method = methods_by_mode[mode]
        except KeyError as e:
            modes_fmtd = ", ".join(map("'{}'".format, methods_by_mode))
            raise ValueError(
                f"invalid format mode '{mode}'; choices: {modes_fmtd}"
            ) from e
        else:
            return method()

    @abc.abstractmethod
    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        ...

    @abc.abstractmethod
    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        ...

    def __repr__(self) -> str:
        """Return a string representation with all instantiation arguments."""
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{key}={sfmt(value)}" for key, value in self.dict().items())
            + ")"
        )

    @staticmethod
    def get_mask_full(trajs: TrajDataset, value: bool = True) -> npt.NDArray[np.bool_]:
        """Get a trajs mask."""
        return np.full(trajs.ds.dims["id"], value, np.bool_)


class AllCriterion(Criterion):
    """Select all trajectories."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return self.get_mask_full(trajs)

    def invert(self) -> InvertedAllCriterion:
        """Invert the criterion."""
        return InvertedAllCriterion()

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "all"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "all"


class InvertedAllCriterion(Criterion):
    """Select no trajectories."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> AllCriterion:
        """Invert the criterion."""
        return AllCriterion()

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "none"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "none"


class _VariableCriterion(Criterion):
    """Base class for ``VariableCriterion`` and its inverse.

    Using a common base class avoid defining their (identical) arguments twice.

    """

    def __init__(
        self,
        variable: str,
        time_idx: int,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Create a new instance."""
        self.variable: str = variable
        self.time_idx: int = time_idx
        self.vmin: Optional[float] = vmin
        self.vmax: Optional[float] = vmax
        self.formatter = VariableCriterionFormatter()

    def dict(self) -> dict[str, Any]:
        """Return dictionary reprentation with all instantiation arguments."""
        return {
            "variable": self.variable,
            "time_idx": self.time_idx,
            "vmin": self.vmin,
            "vmax": self.vmax,
        }


class VariableCriterion(_VariableCriterion):
    """Select trajectories based on a traced variable."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        arr = trajs.get_data(self.variable, idx_time=self.time_idx)
        if n_incomplete := trajs.count(Criteria([LeaveDomainCriterion()])):
            raise NotImplementedError(
                f"{type(self).__name__}.apply for incomplete trajs"
                f" ({n_incomplete:,}/{trajs.count():,})"
            )
        mask = self.get_mask_full(trajs, True)
        if self.vmin is not None:
            mask &= arr >= self.vmin
        if self.vmax is not None:
            mask &= arr <= self.vmax
        return mask

    def invert(self) -> InvertedVariableCriterion:
        """Invert the criterion."""
        other = InvertedVariableCriterion(**self.dict())
        other.formatter = self.formatter
        return other

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return self.formatter.format_human(self)

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return self.formatter.format_file(self)


class InvertedVariableCriterion(_VariableCriterion):
    """Select trajectories that don't meet the trace variable conditions."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> VariableCriterion:
        """Invert the criterion."""
        other = VariableCriterion(**self.dict())
        other.formatter = self.formatter
        return other

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        # pylint: disable=W0212  # protected-access (Criterion._format_human)
        return f"not {self.invert()._format_human()}"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        # pylint: disable=W0212  # protected-access (Criterion._format_file)
        return f"not-{self.invert()._format_file()}"


class LeaveDomainCriterion(Criterion):
    """Select incomplete trajectories that leave the domain."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        mask = (trajs.ds.z.data == trajs.config.nan).sum(axis=0) > 0
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)

    def invert(self) -> InvertedLeaveDomainCriterion:
        """Invert the criterion."""
        return InvertedLeaveDomainCriterion()

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "leaving domain"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "leaving-domain"


class InvertedLeaveDomainCriterion(Criterion):
    """Select complete trajectories that stay inside the domain."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> LeaveDomainCriterion:
        """Invert the criterion."""
        return LeaveDomainCriterion()

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "never leaving domain"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "never-leaving-domain"


class _BoundaryZoneCriterion(Criterion):
    """Base class for invertible ``BoundaryZoneCriterion`` and its inverse.

    Using a common base class avoid defining their (identical) arguments twice.

    """

    def __init__(self, grid: COSMOGridDataset, size_deg: float) -> None:
        """Create a new instance."""
        self.grid: COSMOGridDataset = grid
        self.size_deg: float = size_deg

    def dict(self) -> dict[str, Any]:
        """Return dictionary representation with all instantiation arguments."""
        return {
            "grid": self.grid,
            "size_deg": self.size_deg,
        }


class BoundaryZoneCriterion(_BoundaryZoneCriterion):
    """Select trajectories that enter the domain boundary zone at one point."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        llrlon, urrlon, llrlat, urrlat = self.grid.get_bbox_xy().shrink(self.size_deg)
        rlon, rlat = trajs.ds.longitude.data, trajs.ds.latitude.data
        nan_mask = LeaveDomainCriterion().apply(trajs)
        rlon_mask = np.where(nan_mask, True, (rlon < llrlon) | (rlon > urrlon))
        rlat_mask = np.where(nan_mask, True, (rlat < llrlat) | (rlat > urrlat))
        mask = (rlon_mask | rlat_mask).sum(axis=0) > 0
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)

    def invert(self) -> InvertedBoundaryZoneCriterion:
        """Invert the criterion."""
        return InvertedBoundaryZoneCriterion(**self.dict())

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "in boundary zone"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "in-boundary-zone"


class InvertedBoundaryZoneCriterion(_BoundaryZoneCriterion):
    """Select trajectories that never enter the boundary zone."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> BoundaryZoneCriterion:
        """Invert the criterion."""
        return BoundaryZoneCriterion(**self.dict())

    def _format_human(self) -> str:
        """Format to string suitable for humans (e.g., title)."""
        return "never in boundary zone"

    def _format_file(self) -> str:
        """Format to string suitable for file names."""
        return "never-in-boundary-zone"


@dc.dataclass
class VariableCriterionFormatter:
    """Formatter for variable criterion."""

    units: str = ""
    time: Optional[float] = None
    time_units: str = ""
    time_relative: bool = True
    fmt: str = "{:g}"

    def format_human(self, crit: VariableCriterion) -> str:
        """Format criterion for human consumption."""
        return VariableRangeFormatter(
            name=crit.variable,
            vmin=crit.vmin,
            vmax=crit.vmax,
            **dc.asdict(self),
        ).format_human()

    def format_file(self, crit: VariableCriterion) -> str:
        """Format criterion compatible with file names."""
        return VariableRangeFormatter(
            name=crit.variable,
            vmin=crit.vmin,
            vmax=crit.vmax,
            **dc.asdict(self),
        ).format_file()

    def derive(self, **kwargs: Any) -> VariableCriterionFormatter:
        """Create a derived instance with adapted attributes."""
        return type(self)(**{**dc.asdict(self), **kwargs})


@dc.dataclass
class VariableRangeFormatter:
    """Format a variable with a range given by min. and/or max. value."""

    name: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    units: str = ""
    time: Optional[float] = None
    time_units: str = ""
    time_relative: bool = True
    fmt: str = "{:g}"

    def __post_init__(self) -> None:
        """Finalize instantiation."""
        if (self.vmin, self.vmax) == (None, None):
            raise ValueError("vmin and vmax must not both be None")

    def format_human(self) -> str:
        """Format in human-readable form."""
        units = self.units
        if units:
            units = f" {units}"
        s = self.name + self._format_time()
        lower = "" if self.vmin is None else f"{self.fmt.format(self.vmin)}{units}"
        upper = "" if self.vmax is None else f"{self.fmt.format(self.vmax)}{units}"
        if self.vmin is not None and self.vmax is not None:
            s += f" in {lower} to {upper}"
        elif self.vmin is not None:
            s += f" >= {lower}"
        elif self.vmax is not None:
            s += f" <= {upper}"
        return s

    def format_file(self) -> str:
        """Format compatible with file names."""
        units = self._prepare_units_file(self.units)
        s = self.name.replace(" ", "-") + self._prepare_units_file(
            self._format_time().replace(" ", "")
        )
        lower = "" if self.vmin is None else f"{self.fmt.format(self.vmin)}{units}"
        upper = "" if self.vmax is None else f"{self.fmt.format(self.vmax)}{units}"
        if self.vmin is not None and self.vmax is not None:
            s += f"-{lower}-to-{upper}"
        elif self.vmin is not None:
            s += f"-ge-{lower}"
        elif self.vmax is not None:
            s += f"-le-{upper}"
        assert "/" not in s, f"/ in {s}"  # TODO proper check
        return s.lower()

    @staticmethod
    def _prepare_units_file(units: str) -> str:
        """Prepare units for file-compatible format."""
        units = units.replace(" ", "")
        if "/" in units:
            # TODO more sophisticated approach (regex-based)
            units = (
                units.replace("/s", "s-1")
                .replace("/h", "h-1")
                .replace("/K", "-1")
                .replace("/km", "km-1")
            )
            if "/" in units:
                raise ValueError(f"could not replace all slashes in units: '{units}'")
        return units

    def _format_time(self) -> str:
        """Format time specification following the variable name."""
        if self.time is None:
            return ""
        s = " @ "
        if self.time_relative:
            s += "+"
        s += self.fmt.format(self.time)
        if self.time_units:
            s += f" {self.time_units}"
        return s


# pylint: disable=R0901  # too-many-ancestors (>7)
class Criteria(UserList[Criterion]):
    """A set of combined criteria to select trajectories."""

    def __init__(
        self,
        criteria: Optional[Sequence[Criterion]] = None,
        *,
        require_all: bool = True,
    ) -> None:
        """Create a new instance.

        Args:
            criteria (optional): Individual criteria.

            require_all (optional): Whether selected trajectories must fulfill
                all criteria at once or only one.

        """
        super().__init__(criteria)
        self.require_all: bool = require_all
        self.formatter: CriteriaFormatter = CriteriaFormatter()

    def format(self, mode: str = "human") -> str:
        """Combine all criteria into a single string."""
        return self.formatter.format(self, mode)

    def invert(self) -> Criteria:
        """Invert the criteria."""
        return type(self)(
            criteria=[criterion.invert() for criterion in self],
            require_all=not self.require_all,
        )

    def derive(
        self,
        criteria: Optional[Sequence[Criterion]] = None,
        *,
        require_all: Optional[bool] = None,
    ) -> Criteria:
        """Derive an instance with optionally adapted parameters."""
        if criteria is None:
            criteria = list(self)
        if require_all is None:
            require_all = self.require_all
        return type(self)(criteria, require_all=require_all)

    def dict(self) -> dict[str, Any]:
        """Return dictionary reprentation with all instantiation arguments."""
        return {
            "criteria": list(self),
            "require_all": self.require_all,
        }


@dc.dataclass
class CriteriaFormatter:
    """Format multiple criteria to a single string."""

    times: Optional[Sequence[float]] = None
    vars_attrs: dict[str, Any] = dc.field(default_factory=dict)

    def derive(self, **kwargs: Any) -> CriteriaFormatter:
        """Derive a new instance with adapted attributes."""
        return type(self)(**{**dc.asdict(self), **kwargs})

    def format(self, criteria: Criteria, mode: str) -> str:
        """Format criteria in the given mode."""
        joiners_by_mode = {
            "human": " and " if criteria.require_all else " or ",
            "file": "_and_" if criteria.require_all else "_or_",
        }
        try:
            joiner = joiners_by_mode[mode]
        except KeyError as e:
            modes_fmtd = ", ".join(map("'{}'".format, joiners_by_mode))
            raise ValueError(f"invalid mode '{mode}'; choices: {modes_fmtd}") from e
        return self._format(criteria, mode, joiner)

    def _format(self, criteria: Criteria, mode: str, joiner: str) -> str:
        """Core method to format in a given mode."""
        parts: list[str] = []
        for criterion in criteria:
            if not isinstance(
                criterion, (VariableCriterion, InvertedVariableCriterion)
            ):
                parts.append(criterion.format(mode))
            else:
                old_formatter = criterion.formatter
                # pylint: disable=E1101  # no-member ('Field'.get)
                var_attrs = dict(self.vars_attrs.get(criterion.variable, {}))
                # pylint: disable=E1136  # unsubscriptable-object (self.times)
                if self.times is not None:
                    if "time" not in var_attrs:
                        var_attrs["time"] = self.times[criterion.time_idx]
                criterion.formatter = criterion.formatter.derive(**var_attrs)
                parts.append(criterion.format(mode))
                criterion.formatter = old_formatter
        return joiner.join(parts)
