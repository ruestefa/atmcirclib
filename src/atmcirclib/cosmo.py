"""COSMO output files."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import TYPE_CHECKING

# Third-party
import cartopy.crs as ccrs
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.geo import BoundingBox
from atmcirclib.geo import unrotate_coords
from atmcirclib.typing import PathLike_T

if TYPE_CHECKING:
    # First-party
    from atmcirclib.traj import TrajDataset


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
            ds: xr.Dataset = xr.open_dataset(path, engine="h5netcdf")
        except (IOError, TypeError) as e:
            raise Exception(f"error reading cosmo grid file '{path}'") from e
        return cls(ds, **kwargs)

    @classmethod
    def from_trajs(
        cls,
        trajs: TrajDataset,
        *,
        pole_lon: float = 180.0,
        pole_lat: float = 90.0,
        **kwargs: Any,
    ) -> COSMOGridDataset:
        """Create an instance from a trajectories dataset."""
        rlon_min = np.nanmin(trajs.get_data("longitude"))
        rlon_max = np.nanmax(trajs.get_data("longitude"))
        rlat_min = np.nanmin(trajs.get_data("latitude"))
        rlat_max = np.nanmax(trajs.get_data("latitude"))
        z_max = np.nanmax(trajs.get_data("z"))
        ds = create_cosmo_grid_dataset_ds(
            rlon=np.linspace(rlon_min, rlon_max, 11).astype(np.float32),
            rlat=np.linspace(rlat_min, rlat_max, 11).astype(np.float32),
            pole_rlon=float(pole_lon),
            pole_rlat=float(pole_lat),
            height_toa_m=float(z_max),
        )
        return cls(ds, **kwargs)


def create_cosmo_grid_dataset_ds(
    rlon: npt.NDArray[np.float_],
    rlat: npt.NDArray[np.float_],
    *,
    pole_rlon: float = 180.0,
    pole_rlat: float = 90.0,
    height_toa_m: float = 20_000,
) -> xr.Dataset:
    """Create a dataset representing a COSMO file with grids."""

    def create_coord_time() -> xr.DataArray:
        """Create coordinate variable ``time``."""
        name = "time"
        # mypy 0.941 doesn't recognize return type of np.array (numpy 1.22.3)
        data: npt.NDArray[np.generic] = np.array(
            ["2016-09-20T00:00:00.000000000"], dtype="datetime64[ns]"
        )
        return xr.DataArray(
            name=name,
            data=data,
            dims=(name,),
            coords={name: data},
            attrs={
                "standard_name": "time",
                "long_name": "time",
                "bounds": "time_bnds",
            },
        )

    def create_coord_rlon() -> xr.DataArray:
        """Create coordinate variable ``rlon``."""
        name = "rlon"
        return xr.DataArray(
            name=name,
            data=rlon,
            dims=(name,),
            coords={name: rlon},
            attrs={
                "standard_name": "grid_longitude",
                "long_name": "rotated_longitude",
                "units": "degrees",
            },
        )

    def create_coord_rlat() -> xr.DataArray:
        """Create coordinate variable ``rlat``."""
        name = "rlat"
        return xr.DataArray(
            name=name,
            data=rlat,
            dims=(name,),
            coords={name: rlat},
            attrs={
                "standard_name": "grid_latitude",
                "long_name": "rotated_latitude",
                "units": "degrees",
            },
        )

    def create_coord_lon_lat() -> tuple[xr.DataArray, xr.DataArray]:
        """Create regular variables ``lon`` and ``lat``."""
        rlon = create_coord_rlon()
        rlat = create_coord_rlat()
        rotated_pole = create_rotated_pole()
        lon_arr, lat_arr = unrotate_coords(
            rlon=rlon.data,
            rlat=rlat.data,
            pole_rlon=rotated_pole.attrs["grid_north_pole_longitude"],
            pole_rlat=rotated_pole.attrs["grid_north_pole_latitude"],
            transpose=True,
        )
        dims = ("rlat", "rlon")
        coords = {
            "rlon": rlon.data,
            "rlat": rlat.data,
            "lon": (dims, lon_arr),
            "lat": (dims, lat_arr),
        }
        # mypy 0.941 doesn't recognize return type of xr.DataArray (2022.3.0)
        lon: xr.DataArray = xr.DataArray(
            name="lon",
            data=lon_arr,
            coords=coords,
            dims=dims,
            attrs={
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
            },
        )
        # mypy 0.941 doesn't recognize return type of xr.DataArray (2022.3.0)
        lat: xr.DataArray = xr.DataArray(
            name="lat",
            data=lat_arr,
            coords=coords,
            dims=dims,
            attrs={
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
            },
        )
        return (lon, lat)

    def create_rotated_pole() -> xr.DataArray:
        """Create regular variable ``rotated_pole``."""
        return xr.DataArray(
            name="rotated_pole",
            data=np.array([], dtype="S1"),
            attrs={
                "long_name": "coordinates of the rotated North Pole",
                "grid_mapping_name": "rotated_latitude_latitude",
                "grid_north_pole_latitude": pole_rlat,
                "grid_north_pole_longitude": pole_rlon,
            },
        )

    def create_time_bnds() -> xr.DataArray:
        """Create regular variable ``time_bnds``."""
        time = create_coord_time()
        return xr.DataArray(
            name="time_bnds",
            data=np.array([time.data[0], time.data[-1]]),
            dims=("bnds",),
            attrs={"long_name": "time bounds"},
        )

    def create_height_toa() -> xr.DataArray:
        """Create regular variable ``height_toa``."""
        return xr.DataArray(
            name="height_toa",
            data=np.array(height_toa_m, np.float32),
            attrs={
                "standard_name": "height",
                "long_name": "height of top of model",
                "units": "m",
                "axis": "Z",
                "positive": "up",
            },
        )

    coords: dict[str, xr.DataArray] = {}
    data_vars: dict[str, xr.DataArray] = {}

    coords["time"] = create_coord_time()
    coords["rlon"] = create_coord_rlon()
    coords["rlat"] = create_coord_rlat()
    coords["lon"], coords["lat"] = create_coord_lon_lat()

    data_vars["rotated_pole"] = create_rotated_pole()
    data_vars["time_bnds"] = create_time_bnds()
    data_vars["height_toa"] = create_height_toa()

    attrs = {
        "Conventions": "CF-1.4",
        "conventionsURL": "http://www.cfconventions.org/",
        "creation_date": "2021-12-05 19:03:25",
    }

    return xr.Dataset(
        coords=coords,
        data_vars=data_vars,
        attrs=attrs,
    )
