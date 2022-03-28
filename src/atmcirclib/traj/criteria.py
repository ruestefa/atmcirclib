"""Work with trajectories."""
from __future__ import annotations

# Standard library
import abc
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


class InvertedAllCriterion(Criterion):
    """Select no trajectories."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> AllCriterion:
        """Invert the criterion."""
        return AllCriterion()


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
        return InvertedVariableCriterion(**self.dict())


class InvertedVariableCriterion(_VariableCriterion):
    """Select trajectories that don't meet the trace variable conditions."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> VariableCriterion:
        """Invert the criterion."""
        return VariableCriterion(**self.dict())


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


class InvertedLeaveDomainCriterion(Criterion):
    """Select complete trajectories that stay inside the domain."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> LeaveDomainCriterion:
        """Invert the criterion."""
        return LeaveDomainCriterion()


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


class InvertedBoundaryZoneCriterion(_BoundaryZoneCriterion):
    """Select trajectories that never enter the boundary zone."""

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return ~self.invert().apply(trajs)

    def invert(self) -> BoundaryZoneCriterion:
        """Invert the criterion."""
        return BoundaryZoneCriterion(**self.dict())


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
