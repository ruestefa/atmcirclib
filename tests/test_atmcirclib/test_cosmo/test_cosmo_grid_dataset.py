"""Test class ``COSMOGridDataset``."""
from __future__ import annotations

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.geo import unrotate_coords


def create_grid_xr_dataset() -> xr.Dataset:
    """Create a mock xarray dataset representing a COSMO file with grids."""

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
        data = np.arange(-10.0, 5.1, 1.0)
        return xr.DataArray(
            name=name,
            data=data,
            dims=(name,),
            coords={name: data},
            attrs={
                "standard_name": "grid_longitude",
                "long_name": "rotated_longitude",
                "units": "degrees",
            },
        )

    def create_coord_rlat() -> xr.DataArray:
        """Create coordinate variable ``rlat``."""
        name = "rlat"
        data = np.arange(-6.0, 6.1, 1.0)
        return xr.DataArray(
            name=name,
            data=data,
            dims=(name,),
            coords={name: data},
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
                "grid_north_pole_latitude": 30.0,
                "grid_north_pole_longitude": 178.0,
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

    coords: dict[str, xr.DataArray] = {}
    data_vars: dict[str, xr.DataArray] = {}

    coords["time"] = create_coord_time()
    coords["rlon"] = create_coord_rlon()
    coords["rlat"] = create_coord_rlat()
    coords["lon"], coords["lat"] = create_coord_lon_lat()

    data_vars["rotated_pole"] = create_rotated_pole()
    data_vars["time_bnds"] = create_time_bnds()

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


def test_grid_xr_dataset() -> None:
    """Test initialization of grid xarray test dataset."""
    create_grid_xr_dataset()
