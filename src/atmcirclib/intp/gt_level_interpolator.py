"""Interpolate 3D field to a 2D surface at a given leven.

This module uses gt4py, hence no ``from __future__ import annotations``.

"""

# Standard library
from enum import Enum
from typing import Optional
from typing import Union

# Third-party
import gt4py.gtscript as gts  # type: ignore [import]
import gt4py.storage as gt_store  # type: ignore [import]
import numpy as np
import numpy.typing as npt
from gt4py.gtscript import BACKWARD
from gt4py.gtscript import computation
from gt4py.gtscript import FORWARD
from gt4py.gtscript import interval
from gt4py.gtscript import isnan

# from gt4py.gtscript import PARALLEL
# from gt4py.gtscript import sqrt


class StrEnum(str, Enum):
    """Enum that compares with string values."""

    def __str__(self) -> str:
        """Return value as string."""
        return str(self.value)


class InterpolationDirection(StrEnum):
    """Direction in which vertical interpolation is conducted."""

    up = "up"
    down = "down"


InterpolationDirectionLike_T = Union[InterpolationDirection, str]


class LevelInterpolatorError(Exception):
    """Base exception for error during level interpolation."""


class NonMonotonicGridError(LevelInterpolatorError):
    """Grid is not monotonically in-/decreasing in z."""


class LevelInterpolator:
    """Interpolate 3D fields to 2D surfaces at given levels."""

    def __init__(
        self,
        grid: npt.ArrayLike,
        *,
        direction: Optional[InterpolationDirectionLike_T] = None,
        dtype: npt.DTypeLike = np.float32,
        gt_backend: str = "gtc:numpy",
    ) -> None:
        """Create a new instance.

        Args:
            grid: 3D grid array, ideally with monotonically increasing or
                decreasing values.

            direction (optional): Direction in which vertical interpolation
                is conducted; typically 'forward'/'backward' for monotonically
                in-/decreasing grids in z.

            dtype (optional): Data type used internally during computations.

            gt_backend (optional): Backend used by GT4Py.

        """
        self.grid: npt.NDArray[np.float_] = np.asarray(grid, dtype)
        if direction is None:
            try:
                direction = self._derive_direction()
            except NonMonotonicGridError as e:
                raise ValueError(
                    "must specify direction for grid that is not non-monotonically"
                    " in-/decreasing grid in z"
                ) from e
        self.direction: InterpolationDirection = InterpolationDirection(direction)
        self.dtype: npt.DTypeLike = dtype
        self.gt_backend: str = gt_backend

    def to_level(self, fld: npt.ArrayLike, lvl: float) -> npt.NDArray[np.float_]:
        """Interpolate to a single level."""
        lvl = float(lvl)  # gt4py doesn't accept ints in place of floats
        dtype_in = fld.dtype if isinstance(fld, np.ndarray) else self.dtype
        fld = np.asarray(fld, self.dtype)
        DTYPE = self.dtype

        @gts.function
        def gtf_intp_point(
            fld: gts.Field[gts.IJK, DTYPE], grid: gts.Field[gts.IJK, DTYPE], lvl: float
        ) -> gts.Field[gts.IJK, DTYPE]:
            """Perform interpolation of a point between two levels."""
            return fld[0, 0, 0] + (
                (lvl - grid[0, 0, 0])
                / (grid[0, 0, 1] - grid[0, 0, 0])
                * (fld[0, 0, 1] - fld[0, 0, 0])
            )

        @gts.stencil(backend=self.gt_backend)
        def gts_intp_up(
            fld: gts.Field[gts.IJK, DTYPE],
            grid: gts.Field[gts.IJK, DTYPE],
            lvl: float,
            intp: gts.Field[gts.IJ, DTYPE],
            vnan: float,
        ) -> None:
            """Perform interpolation upward."""
            with computation(FORWARD), interval(...):
                intp[...] = vnan
            with computation(FORWARD), interval(...):
                if (grid[0, 0, 0] <= lvl and grid[0, 0, 1] >= lvl) or (
                    grid[0, 0, 0] >= lvl and grid[0, 0, 1] <= lvl
                ):
                    if isnan(intp[0, 0]) or intp[0, 0] == vnan:
                        # Note: As of 2022-04-07, assigning the resulf of the
                        # function call directly to intp triggers an error in
                        # `gtscript_frontend.py`, thus the addition of 0.0.
                        intp[...] = 0.0 + gtf_intp_point(fld, grid, lvl)

        @gts.stencil(backend=self.gt_backend)
        def gts_intp_down(
            fld: gts.Field[gts.IJK, DTYPE],
            grid: gts.Field[gts.IJK, DTYPE],
            lvl: float,
            intp: gts.Field[gts.IJ, DTYPE],
            vnan: float,
        ) -> None:
            """Perform interpolation downward."""
            with computation(BACKWARD), interval(...):
                intp[...] = vnan
            with computation(BACKWARD), interval(...):
                if (grid[0, 0, 0] <= lvl and grid[0, 0, 1] >= lvl) or (
                    grid[0, 0, 0] >= lvl and grid[0, 0, 1] <= lvl
                ):
                    if isnan(intp[0, 0]) or intp[0, 0] == vnan:
                        # Note: As of 2022-04-07, assigning the resulf of the
                        # function call directly to intp triggers an error in
                        # `gtscript_frontend.py`, thus the addition of 0.0.
                        intp[...] = 0.0 + gtf_intp_point(fld, grid, lvl)

        # Define stores
        shape2d = fld.shape[:2]
        fld_store = gt_store.from_array(
            fld,
            default_origin=(0, 0, 0),
            dtype=self.dtype,
            backend=self.gt_backend,
        )
        grid_store = gt_store.from_array(
            self.grid,
            default_origin=(0, 0, 0),
            dtype=self.dtype,
            backend=self.gt_backend,
        )
        intp_store = gt_store.empty(
            shape=shape2d,
            mask="IJ",
            default_origin=(0, 0),
            dtype=self.dtype,
            backend=self.gt_backend,
        )

        if self.direction == "up":
            gts_intp_up(fld_store, grid_store, lvl, intp_store, np.nan)
        elif self.direction == "down":
            gts_intp_down(fld_store, grid_store, lvl, intp_store, np.nan)
        else:
            raise ValueError(f"invalid direction '{self.direction}'")

        return np.asarray(intp_store.data, dtype_in)

    def to_levels(
        self, fld: npt.ArrayLike, lvls: npt.ArrayLike
    ) -> npt.NDArray[np.float_]:
        """Interpolate to multiple levels."""
        lvls = np.asarray(lvls, self.dtype)
        if len(lvls.shape) != 1:
            raise ValueError(
                f"lvls must be a one-dimensional array-like, but has shape {lvls.shape}"
            )
        layers: list[npt.NDArray[np.float_]] = []
        for lvl in lvls:
            layers.append(self.to_level(fld, lvl))
        return np.array(layers)

    def _derive_direction(self) -> InterpolationDirection:
        """Derive interpol. direction from monotonically in-/decreasing grid."""
        deltas = self.grid[:, :, 1:] - self.grid[:, :, :-1]
        if (deltas > 0).all():
            return InterpolationDirection.up
        elif (deltas < 0).all():
            return InterpolationDirection.down
        else:
            raise NonMonotonicGridError(
                "cannot derive direction from non-monotonic grid in z"
            )
