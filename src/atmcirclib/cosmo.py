"""COSMO output files."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import cast

# Third-party
import cartopy.crs as ccrs
import xarray as xr

# First-party
from atmcirclib.geo import BoundingBox
from atmcirclib.typing import PathLike_T


class COSMOGridDataset:
    """File with grid information of COSMO simulation."""

    def __init__(self, ds: xr.Dataset, z_unit: str = "m") -> None:
        """Create new instance."""
        self.ds: xr.Dataset = ds
        self.z_unit: str = z_unit

    def get_bbox_xy(self) -> BoundingBox:
        """Get (lon, lat) bounding box."""
        return BoundingBox.from_coords(self.ds.rlon.data, self.ds.rlat.data)

    def get_bbox_xz(self) -> BoundingBox:
        """Get (lon, z) bounding box."""
        bbox_xy = self.get_bbox_xy()
        zmin = 0.0
        zmax = float(self.ds.height_toa.data)
        return BoundingBox(
            llx=bbox_xy.llx,
            urx=bbox_xy.urx,
            lly=self._scale_z(zmin),
            ury=self._scale_z(zmax),
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

    def _scale_z(self, val: float) -> float:
        """Scale height value to desired unit ``self.z_unit``."""
        unit = self.ds.height_toa.attrs["units"]
        if self.z_unit == unit:
            fact = 1.0
        elif self.z_unit == "km" and unit == "m":
            fact = 1 / 1000.0
        elif self.z_unit == "m" and unit == "km":
            fact = 1000.0
        else:
            raise Exception(
                f"unknown target z unit '{self.z_unit}' or input z unit '{unit}'"
            )
        return fact * val

    @classmethod
    def from_file(cls, path: PathLike_T, **kwargs: Any) -> COSMOGridDataset:
        """Create an instance from a NetCDF file."""
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        try:
            ds: xr.Dataset = cast(xr.Dataset, xr.open_dataset(path, engine="h5netcdf"))
        except (IOError, TypeError) as e:
            raise Exception(f"error reading cosmo grid file '{path}'") from e
        return cls(ds, **kwargs)
