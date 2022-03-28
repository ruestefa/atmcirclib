"""Work with trajectories."""
from __future__ import annotations

# Standard library
import abc
import dataclasses as dc
from collections.abc import Collection
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

# Custom types
Criteria_T = Collection["Criterion"]


class Criterion(abc.ABC):
    """Base class of criteria to selection trajectories."""

    @abc.abstractmethod
    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        return self.get_mask_full(trajs)

    @staticmethod
    def get_mask_full(trajs: TrajDataset, value: bool = True) -> npt.NDArray[np.bool_]:
        """Get a trajs mask."""
        return np.full(trajs.ds.dims["id"], value, np.bool_)


@dc.dataclass
class VariableCriterion(Criterion):
    """Select trajectories based on a traced variable."""

    variable: str
    time_idx: int
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        arr = trajs.get_data(self.variable, idx_time=self.time_idx)
        if n_incomplete := trajs.count([IncompleteCriterion()]):
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


# TODO eliminate/improve
@dc.dataclass
class IncompleteCriterion(Criterion):
    """Select incomplete trajectories that leave the domain."""

    value: bool = True

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        mask = (trajs.ds.z.data == trajs.config.nan).sum(axis=0) > 0
        if not self.value:
            mask = ~mask
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)


# TODO eliminate/improve
@dc.dataclass
class BoundaryZoneCriterion(Criterion):
    """Select trajectories that enter the domain boundary zone at one point."""

    grid: COSMOGridDataset
    size_deg: float
    value: bool = True

    def apply(self, trajs: TrajDataset) -> npt.NDArray[np.bool_]:
        """Apply criterion to trajectories and return 1D mask array."""
        llrlon, urrlon, llrlat, urrlat = self.grid.get_bbox_xy().shrink(self.size_deg)
        rlon, rlat = trajs.ds.longitude.data, trajs.ds.latitude.data
        nan_mask = IncompleteCriterion().apply(trajs)
        rlon_mask = np.where(nan_mask, True, (rlon < llrlon) | (rlon > urrlon))
        rlat_mask = np.where(nan_mask, True, (rlat < llrlat) | (rlat > urrlat))
        mask = (rlon_mask | rlat_mask).sum(axis=0) > 0
        if not self.value:
            mask = ~mask
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)
