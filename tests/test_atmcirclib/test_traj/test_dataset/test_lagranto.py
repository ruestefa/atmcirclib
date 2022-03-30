"""Initialize from a LAGRANTO NetCDF output file (differ from COSMO format)."""
from __future__ import annotations

# Standard library
from typing import Any

# Third-party
import cartopy.crs as ccrs
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.traj.lagranto import convert_traj_ds_lagranto_to_cosmo

# Local
from .shared import TrajDatasetDsFactory

# pylint: disable=R0201  # no-self-use

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Input dataset as read from LAGRANTO output file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

LAGRANTO_RAW_DATA_D: dict[str, list[list[float]]] = {}
LAGRANTO_SCALE_FACT_D: dict[str, float] = {}
LAGRANTO_DTYPE_D: dict[str, npt.DTypeLike] = {}
LAGRANTO_ATTRS_D: dict[str, dict[str, str]] = {}
LAGRANTO_VNAN_D: dict[str, float] = {}

LAGRANTO_DIMS: tuple[str, str] = ("ntim", "ntra")
LAGRANTO_VNAN: float = -999.0
LAGRANTO_IN_ATTRS: dict[str, Any] = {
    "ref_year": 2016,
    "ref_month": 9,
    "ref_day": 21,
    "ref_hour": 12,
    "ref_min": 0,
    "duration": -210,
    "pollon": 0.0,
    "pollat": 90.0,
}

_name = "time"
LAGRANTO_ATTRS_D[_name] = {}
# Backward trajs, therefore negative time
# Format appears to be '<hour>.<10min>'
LAGRANTO_RAW_DATA_D[_name] = [
    [-0.0, -0.0, -0.0, -0.0],
    [-0.3, -0.3, -0.3, -0.3],
    [-1.0, -1.0, -1.0, -1.0],
    [-1.3, -1.3, -1.3, -1.3],
    [-2.0, -2.0, -2.0, -2.0],
    [-2.3, -2.3, -2.3, -2.3],
    [-3.0, -3.0, -3.0, -3.0],
    [-3.3, -3.3, -3.3, -3.3],
]

_name = "lon"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_VNAN_D[_name] = _nan_ = 167.615  # weird missing value; -999 rotated?
# Hand-written
LAGRANTO_RAW_DATA_D[_name] = [
    [-43.5, -21.0, -30.5, -10.0],
    [-44.0, -20.0, -30.5, -10.0],
    [-44.5, -19.0, -30.5, -10.0],
    [-45.0, -18.0, -31.0, -10.5],
    [_nan_, -17.0, -32.0, -11.0],
    [_nan_, -16.0, -32.5, -12.5],
    [_nan_, -15.0, -32.5, -14.5],
    [_nan_, -14.0, -32.5, -17.0],
]

# rlon:
#   [-17.8, -14.8, -24.7, -4.71],
#   [-17.9, -14.1, -24.7, -5.02],
#   [-17.9, -13.3, -24.0, -4.05],
#   [-17.9, -12.6, -24.2, -4.49],
#   [+81.0, -11.9, -25.4, -4.89],
#   [+81.0, -11.1, -25.5, -5.78],
#   [+81.0, -10.3, -26.3, -6.86],
#   [+81.0, -9.62, -26.3, -8.22],
# -> Note: -999 % 360 = 81

_name = "lat"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_VNAN_D[_name] = _nan_ = 30.9995  # weird missing value; -999 rotated?
# Hand-written
LAGRANTO_RAW_DATA_D[_name] = [
    [+62.7, +41.1, +34.5, +54.0],
    [+63.0, +41.0, +34.5, +51.5],
    [+63.3, +40.9, +36.0, +59.5],
    [+63.6, +40.8, +36.5, +58.0],
    [_nan_, +40.7, +35.5, +57.0],
    [_nan_, +40.6, +36.0, +56.5],
    [_nan_, +40.5, +34.5, +56.5],
    [_nan_, +40.4, +34.5, +56.5],
]

# rlat:
#   [+8.44, -16.7, -20.1, -5.71],
#   [+8.81, -17.0, -20.1, -8.19],
#   [+9.19, -17.3, -18.7, -0.25],
#   [+9.56, -17.6, -18.1, -1.71],
#   [+80.9, -17.9, -18.6, -2.66],
#   [+80.9, -18.2, -18.0, -3.04],
#   [+80.9, -18.4, -19.3, -2.84],
#   [+80.9, -18.7, -19.3, -2.56],
# -> Note: -999 % 180 = 81

_name = "z"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_SCALE_FACT_D[_name] = 1000  # km => m
_nan_ = LAGRANTO_VNAN
# Hand-written
LAGRANTO_RAW_DATA_D[_name] = [
    [9.850, 3.180, 0.900, 7.400],
    [9.800, 3.200, 1.000, 7.200],
    [9.750, 3.220, 1.200, 7.000],
    [9.700, 3.240, 1.600, 6.800],
    [_nan_, 3.260, 2.000, 6.600],
    [_nan_, 3.280, 2.200, 6.400],
    [_nan_, 3.300, 2.400, 6.200],
    [_nan_, 3.320, 2.500, 6.000],
]

_name = "P"
LAGRANTO_ATTRS_D[_name] = {}
_nan_ = LAGRANTO_VNAN
# Derived from z with calculator
# (https://keisan.casio.com/exec/system/1224562962)
LAGRANTO_RAW_DATA_D[_name] = [
    [278.6, 690.7, 911.7, 396.1],
    [280.7, 689.0, 900.0, 407.3],
    [282.8, 687.2, 879.7, 418.6],
    [284.9, 685.5, 838.5, 430.3],
    [_nan_, 683.8, 798.9, 442.2],
    [_nan_, 682.1, 779.6, 454.4],
    [_nan_, 680.4, 760.8, 466.8],
    [_nan_, 678.7, 751.5, 479.5],
]

_name = "T"
LAGRANTO_ATTRS_D[_name] = {}
_nan_ = LAGRANTO_VNAN
# Derived from z with calculator
# (https://keisan.casio.com/exec/system/1224562962)
LAGRANTO_RAW_DATA_D[_name] = [
    [-43.2, +0.18, +15.0, -27.3],
    [-42.9, +0.05, +14.4, -26.0],
    [-42.5, -0.08, +13.1, -24.7],
    [-42.2, -0.21, +10.5, -23.4],
    [_nan_, -0.34, +7.85, -22.1],
    [_nan_, -0.47, +6.55, -20.8],
    [_nan_, -0.60, +5.25, -19.5],
    [_nan_, -0.73, +4.60, -18.5],
]

_name = "U"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_SCALE_FACT_D[_name] = 1000 / 3600  # km/h => m/s
_nan_ = LAGRANTO_VNAN
# Approximated from rlon; in km/h
LAGRANTO_RAW_DATA_D[_name] = [
    [-2.30, +160.0, +0.00, -67.3],
    [-1.77, +160.0, +76.7, +73.2],
    [-0.70, +162.0, +60.1, +58.0],
    [-0.16, +163.0, -152.0, -92.7],
    [_nan_, +165.0, -151.0, -142.0],
    [_nan_, +166.0, -96.0, -217.0],
    [_nan_, +168.0, -80.8, -268.0],
    [_nan_, +169.0, -0.00, -297.0],
]

_name = "V"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_SCALE_FACT_D[_name] = 1000 / 3600  # km/h => m/s
_nan_ = LAGRANTO_VNAN
# Approximated from rlat; in km/h
LAGRANTO_RAW_DATA_D[_name] = [
    [+82.9, -68.7, +0.00, -546.0],
    [+82.7, -67.6, +148.0, +600.0],
    [+82.5, -65.3, +217.0, +713.0],
    [+82.3, -63.0, +10.6, -265.0],
    [_nan_, -60.7, +11.1, -146.0],
    [_nan_, -58.3, -76.8, -20.1],
    [_nan_, -55.9, -146.0, +52.3],
    [_nan_, -54.7, -0.00, +62.6],
]

_name = "W"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_SCALE_FACT_D[_name] = 1e-2
_nan_ = LAGRANTO_VNAN
# Approximated from z; in 100 * m/s
LAGRANTO_RAW_DATA_D[_name] = [
    [-2.78, +1.11, +5.56, -11.1],
    [-2.78, +1.11, +8.33, -11.1],
    [-2.78, +1.11, +16.7, -11.1],
    [-2.78, +1.11, +22.2, -11.1],
    [_nan_, +1.11, +16.7, -11.1],
    [_nan_, +1.11, +11.1, -11.1],
    [_nan_, +1.11, +8.33, -11.1],
    [_nan_, +1.11, +5.56, -11.1],
]

_name = "POT_VORTIC"
LAGRANTO_ATTRS_D[_name] = {}
LAGRANTO_SCALE_FACT_D[_name] = 1e-6
_nan_ = LAGRANTO_VNAN
# Approximated from z; in pvu (1e-6 m2 s-1 K kg-1)
LAGRANTO_RAW_DATA_D[_name] = [
    [+2.50, +0.50, +0.30, +0.40],
    [+2.50, +0.50, +0.30, +0.40],
    [+2.50, +0.50, +0.30, +0.40],
    [+2.50, +0.50, +0.30, +0.40],
    [_nan_, +0.50, +0.30, +0.40],
    [_nan_, +0.50, +0.30, +0.40],
    [_nan_, +0.50, +0.30, +0.40],
    [_nan_, +0.50, +0.30, +0.40],
]

lagranto_ds_factory = TrajDatasetDsFactory(
    attrs=LAGRANTO_IN_ATTRS,
    dims=LAGRANTO_DIMS,
    data_d=LAGRANTO_RAW_DATA_D,
    attrs_d=LAGRANTO_ATTRS_D,
    dtype_d=LAGRANTO_DTYPE_D,
    scale_fact_d=LAGRANTO_SCALE_FACT_D,
    vnan=LAGRANTO_VNAN,
)
create_lagranto_ds = lagranto_ds_factory.run

LAGRANTO_REF_DATA_D = lagranto_ds_factory.data_d

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reference dataset as would be read from COSMO output file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Specification of rotated grid on which computation took place
# (LAGRANTO rotated coordinats to regulat lon/lat before output)
ROT_POLE_LON = 178.0
ROT_POLE_LAT = 30.0
# ROT_LON_LIM = (-25.0, +11.0)
# ROT_LAT_LIM = (-14.7, +9.7)

# Rotate raw lon/lat points and replace missing values by NaNs
_mask = np.array(LAGRANTO_RAW_DATA_D["z"]) == LAGRANTO_VNAN
_lon = np.where(_mask, np.nan, np.array(LAGRANTO_RAW_DATA_D["lon"]))
_lat = np.where(_mask, np.nan, np.array(LAGRANTO_RAW_DATA_D["lat"]))
REF_RLON, REF_RLAT, _ = np.moveaxis(
    # pylint: disable=E0110  # abstract-class-instantiated (ccrs.*)
    ccrs.RotatedPole(
        pole_longitude=ROT_POLE_LON, pole_latitude=ROT_POLE_LAT
    ).transform_points(ccrs.PlateCarree(), _lon, _lat),
    2,
    0,
)

COSMO_RAW_COORDS_D: dict[str, list[float]] = {}
COSMO_RAW_DATA_D: dict[str, list[list[float]]] = {}
COSMO_SCALE_FACT_D: dict[str, float] = {}
COSMO_DTYPE_D: dict[str, npt.DTypeLike] = {}
COSMO_ATTRS_D: dict[str, dict[str, str]] = {}
COSMO_VNAN_D: dict[str, float] = {}

COSMO_DIMS: tuple[str, str] = ("time", "id")
COSMO_VNAN: float = -999.0
COSMO_IN_ATTRS: dict[str, Any] = {
    "ref_year": 2016,
    "ref_month": 9,
    "ref_day": 21,
    "ref_hour": 12,
    "ref_min": 0,
    "ref_sec": 0,
    "duration_in_sec": -12600.0,
    "pollon": ROT_POLE_LON,
    "pollat": ROT_POLE_LAT,
    "output_timestep_in_sec": -1800.0,
}

_name = "time"
COSMO_ATTRS_D[_name] = {
    "standard_name": "time",
    "long_name": "time",
}
COSMO_DTYPE_D[_name] = "timedelta64[m]"
COSMO_RAW_COORDS_D[_name] = [-0, -30, -60, -90, -120, -150, -180, -210]

RAW_RLON = np.where(np.isnan(REF_RLON), COSMO_VNAN, REF_RLON).tolist()
RAW_RLAT = np.where(np.isnan(REF_RLAT), COSMO_VNAN, REF_RLAT).tolist()

_name = "longitude"
COSMO_ATTRS_D[_name] = {
    "standard_name": "grid_longitude",
    "long_name": "rotated longitudes",
    "units": "degrees",
}
COSMO_RAW_DATA_D[_name] = RAW_RLON

_name = "latitude"
COSMO_ATTRS_D[_name] = {
    "standard_name": "grid_latitude",
    "long_name": "rotated latitudes",
    "units": "degrees",
}
COSMO_RAW_DATA_D[_name] = RAW_RLAT

_name = "z"
COSMO_ATTRS_D[_name] = {
    "standard_name": "height",
    "long_name": "height above mean sea level",
    "units": "m AMSL",
}
COSMO_SCALE_FACT_D[_name] = LAGRANTO_SCALE_FACT_D[_name]
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "P"
COSMO_ATTRS_D[_name] = {
    "standard_name": "air_pressure",
    "long_name": "pressure",
    "units": "Pa",
}
COSMO_SCALE_FACT_D[_name] = 100  # hPa => Pa
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "T"
COSMO_ATTRS_D[_name] = {
    "standard_name": "air_temperature",
    "long_name": "temperature",
    "units": "K",
}
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "U"
COSMO_ATTRS_D[_name] = {
    "standard_name": "grid_eastward_wind",
    "long_name": "U-component of wind",
    "units": "m s-1",
}
COSMO_SCALE_FACT_D[_name] = LAGRANTO_SCALE_FACT_D[_name]
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "V"
COSMO_ATTRS_D[_name] = {
    "standard_name": "grid_northward_wind",
    "long_name": "V-component of wind",
    "units": "m s-1",
}
COSMO_SCALE_FACT_D[_name] = LAGRANTO_SCALE_FACT_D[_name]
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "W"
COSMO_SCALE_FACT_D[_name] = LAGRANTO_SCALE_FACT_D[_name]
COSMO_ATTRS_D[_name] = {
    "standard_name": "upward_air_velocity",
    "long_name": "vertical wind velocity",
    "units": "m s-1",
}
COSMO_SCALE_FACT_D[_name] = 1 / 3.6  # km/h -> m/s
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

_name = "POT_VORTIC"
COSMO_ATTRS_D[_name] = {
    "standard_name": "ertel_potential_vorticity",
    "long_name": "potential vorticity",
    "units": "K m2 kg-1 s-1",
}
COSMO_SCALE_FACT_D[_name] = LAGRANTO_SCALE_FACT_D[_name]
COSMO_RAW_DATA_D[_name] = LAGRANTO_RAW_DATA_D[_name]

cosmo_ds_factory = TrajDatasetDsFactory(
    attrs=COSMO_IN_ATTRS,
    dims=COSMO_DIMS,
    coords_d=COSMO_RAW_COORDS_D,
    data_d=COSMO_RAW_DATA_D,
    attrs_d=COSMO_ATTRS_D,
    dtype_d=COSMO_DTYPE_D,
    scale_fact_d=COSMO_SCALE_FACT_D,
    vnan=COSMO_VNAN,
)
create_cosmo_ds = cosmo_ds_factory.run

COSMO_REF_DATA_D = cosmo_ds_factory.data_d
COSMO_REF_COORDS_D = cosmo_ds_factory.coords_d


class Test_TestLagrantoDataset:
    """Test the LAGRANTO test dataset."""

    def test_time(self) -> None:
        """Test the variable 'time'."""
        ds = create_lagranto_ds()
        assert "time" in ds.variables
        assert np.allclose(ds.time.data, LAGRANTO_REF_DATA_D["time"])


class Test_ConvertLagrantoToCosmo:
    """Convert the LAGRANTO NetCDF output file to COSMO output format."""

    @property
    def ds_lagra(self) -> xr.Dataset:
        """Create a LAGRANTO dataset."""
        return convert_traj_ds_lagranto_to_cosmo(
            ds=create_lagranto_ds(),
            pole_lon=ROT_POLE_LON,
            pole_lat=ROT_POLE_LAT,
        )

    @property
    def ds_cosmo(self) -> xr.Dataset:
        """Create a COSMO dataset."""
        return create_cosmo_ds()

    def compare_vars(self, v1: xr.DataArray, v2: xr.DataArray, name: str = "") -> None:
        """Compare two variables."""
        assert v1.name == v2.name
        if name:
            assert v1.name == name
        assert v1.dims == v2.dims
        assert v1.dtype == v2.dtype
        try:
            assert np.allclose(v1.data, v2.data)
        except TypeError:
            assert np.array_equal(v1.data, v2.data)
        assert v1.attrs == v2.attrs

    def test_time(self) -> None:
        """Compare coordinate variable 'time'."""
        self.compare_vars(self.ds_lagra.time, self.ds_cosmo.time, "time")

    def test_coords(self) -> None:
        """Compare coordinate variables."""
        coords_lagra = self.ds_lagra.coords
        coords_cosmo = self.ds_lagra.coords
        assert coords_lagra.keys() == coords_cosmo.keys()
        for name, coord_lagra in coords_lagra.items():
            self.compare_vars(coord_lagra, coords_cosmo[name])

    def test_lon_lat(self) -> None:
        """Compare lon/lat in back-rotated coordinates."""
        self.compare_vars(self.ds_lagra.longitude, self.ds_cosmo.longitude)
        self.compare_vars(self.ds_lagra.latitude, self.ds_cosmo.latitude)

    def test_lon_lat_norot(self) -> None:
        """Compare lon/lat in non-rotated coordinates."""
        ds_lagra = convert_traj_ds_lagranto_to_cosmo(ds=create_lagranto_ds())
        mask = np.array(LAGRANTO_RAW_DATA_D["z"]) == LAGRANTO_VNAN
        lon = np.where(mask, LAGRANTO_VNAN, np.array(LAGRANTO_RAW_DATA_D["lon"]))
        lat = np.where(mask, LAGRANTO_VNAN, np.array(LAGRANTO_RAW_DATA_D["lat"]))
        assert np.allclose(ds_lagra.longitude.data, lon)
        assert np.allclose(ds_lagra.latitude.data, lat)

    def test_attrs(self) -> None:
        """Compare global dataset attributes."""
        attrs_lagra = dict(self.ds_lagra.attrs)
        attrs_cosmo = dict(self.ds_cosmo.attrs)
        attrs_lagra.pop("_orig")
        assert attrs_lagra == attrs_cosmo
