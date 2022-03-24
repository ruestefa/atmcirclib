"""Geometric and geographic utilities."""
from __future__ import annotations

# Standard library
from collections.abc import Sequence
from typing import cast
from typing import NamedTuple
from typing import Optional
from typing import Union

# Third-party
import cartopy.crs as ccrs
import numpy as np
import numpy.typing as npt


class BoundingBox(NamedTuple):
    """A rectangular bounding box."""

    llx: float
    urx: float
    lly: float
    ury: float

    def get_center(self) -> tuple[float, float]:
        """Get the center lon/lat coordinates."""
        return (0.5 * (self.llx + self.urx), 0.5 * (self.lly + self.ury))

    def get_width(self) -> float:
        """Get width."""
        return self.urx - self.llx

    def get_height(self) -> float:
        """Get height."""
        return self.ury - self.lly

    def get_aspect(self) -> float:
        """Get aspect ratio."""
        return self.get_width() / self.get_height()

    def get_xlim(self) -> tuple[float, float]:
        """Get limits in x direction."""
        return (self.llx, self.urx)

    def get_ylim(self) -> tuple[float, float]:
        """Get limits in y direction."""
        return (self.lly, self.ury)

    def shrink(self, bnd: Union[float, tuple[float, float]]) -> BoundingBox:
        """Return a copy shrunk by ``bnd`` from the outside in.

        Args:
            bnd: Either a single value that is applied to all four boundaries,
                or separate values ``(dx, dy)``, whereby the domain is shrunk
                from the left and right by ``dx`` and from the bottom and top by
                ``dy``.

        """
        try:
            bnd_x = bnd_y = float(cast(float, bnd))
        except TypeError:
            try:
                bnd_x, bnd_y = map(float, cast(tuple[float, float], bnd))
            except TypeError as e:
                raise ValueError(
                    f"bnd must be a float or a pair of floats, not {bnd}"
                ) from e
        llx = self.llx + bnd_x
        urx = self.urx - bnd_x
        lly = self.lly + bnd_y
        ury = self.ury - bnd_y
        if llx > urx:
            llx = urx = 0.5 * (llx + urx)
        if lly > ury:
            lly = ury = 0.5 * (lly + ury)
        return type(self)(llx=llx, urx=urx, lly=lly, ury=ury)

    def swapaxes(self) -> BoundingBox:
        """Return a copy with swapped x and y axes."""
        return type(self)(llx=self.lly, urx=self.ury, lly=self.llx, ury=self.urx)

    def derive(
        self,
        *,
        llx: Optional[float] = None,
        urx: Optional[float] = None,
        lly: Optional[float] = None,
        ury: Optional[float] = None,
    ) -> BoundingBox:
        """Derive a copy with adapted parameters."""
        return type(self)(
            llx=self.llx if llx is None else llx,
            urx=self.urx if urx is None else urx,
            lly=self.lly if lly is None else lly,
            ury=self.ury if ury is None else ury,
        )

    @classmethod
    def from_coords(
        cls,
        xs: Union[Sequence[float], npt.NDArray[np.float_]],
        ys: Union[Sequence[float], npt.NDArray[np.float_]],
    ) -> BoundingBox:
        """Create a new instance from coordinate arrays."""
        return cls(llx=xs[0], urx=xs[-1], lly=ys[0], ury=ys[-1])


def unrotate_coords(
    rlon: npt.NDArray[np.float_],
    rlat: npt.NDArray[np.float_],
    pole_rlon: float,
    pole_rlat: float,
    transpose: bool = False,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Turn 1D rotated lon/lat coordinates into 2D regular lon/lat arrays."""
    # pylint: disable=E0110  # abstract-class-instantiated (RotatedPole, PlateCarree)
    proj_rot = ccrs.RotatedPole(pole_longitude=pole_rlon, pole_latitude=pole_rlat)
    proj_geo = ccrs.PlateCarree()
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)
    lon, lat, _ = np.transpose(proj_geo.transform_points(proj_rot, rlon2d, rlat2d))
    if transpose:
        lon = lon.T
        lat = lat.T
    return (lon, lat)
