"""COSMO outout files."""
from __future__ import annotations

# Standard library
from typing import cast

# Third-party
import cartopy.crs as ccrs
import xarray as xr

# First-party
from atmcirclib.geo import BoundingBox
from atmcirclib.typing import PathLike_T


class COSMOGridDataset:
    """File with grid information of COSMO simulation."""

    def __init__(self, ds: xr.Dataset) -> None:
        """Create new instance."""
        self.ds: xr.Dataset = ds

    def get_bbox_xy(self) -> BoundingBox:
        """Get (lon, lat) bounding box."""
        return BoundingBox.from_coords(self.ds.rlon.data, self.ds.rlat.data)

    def get_bbox_xz(self) -> BoundingBox:
        """Get (lon, z) bounding box."""
        bbox_xy = self.get_bbox_xy()
        zmin = 0.0
        zmax = float(self.ds.height_toa.data)
        zmax /= 1000.0  # m => km  # TODO make this more explicit
        return BoundingBox(
            llx=bbox_xy.llx,
            urx=bbox_xy.urx,
            lly=zmin,
            ury=zmax,
        )

    def get_bbox_yz(self) -> BoundingBox:
        """Get (lat, z) bounding box."""
        bbox_xy = self.get_bbox_xy()
        bbox_xz = self.get_bbox_xz()
        return BoundingBox(
            llx=bbox_xy.lly,
            urx=bbox_xy.ury,
            lly=bbox_xz.lly,
            ury=bbox_xz.ury,
        )

    def get_proj(self) -> ccrs.Projection:
        """Get domain projection."""
        # pylint: disable=E0110  # abstract-class-instantiated (RotatedPole)
        return ccrs.RotatedPole(
            pole_latitude=self.ds.rotated_pole.grid_north_pole_latitude,
            pole_longitude=self.ds.rotated_pole.grid_north_pole_longitude,
        )

    @classmethod
    def from_file(cls, path: PathLike_T) -> COSMOGridDataset:
        """Create an instance from a NetCDF file."""
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        ds: xr.Dataset = cast(xr.Dataset, xr.open_dataset(path))
        return cls(ds)
