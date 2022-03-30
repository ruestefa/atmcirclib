"""Handle output files of LAGRANTO, like conversion to COSMO output format."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import cast

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

__all__: list[str] = [
    "convert_traj_ds_lagranto_to_cosmo",
]


def convert_traj_ds_lagranto_to_cosmo(ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
    """Convert a LAGRANTO output dataset to COSMO output format."""
    return TrajDsLagrantoToCosmoConverter(**kwargs).convert(ds)


class TrajDsLagrantoToCosmoConverter:
    """Convert a LAGRANTO output dataset to COSMO output format."""

    var_attrs_d: dict[str, dict[str, Any]] = {
        "time": {
            "standard_name": "time",
            "long_name": "time",
        },
        "longitude": {
            "standard_name": "grid_longitude",
            "long_name": "rotated longitudes",
            "units": "degrees",
        },
        "latitude": {
            "standard_name": "grid_latitude",
            "long_name": "rotated latitudes",
            "units": "degrees",
        },
        "z": {
            "standard_name": "height",
            "long_name": "height above mean sea level",
            "units": "m AMSL",
        },
        "P": {
            "standard_name": "air_pressure",
            "long_name": "pressure",
            "units": "Pa",
        },
        "T": {
            "standard_name": "air_temperature",
            "long_name": "temperature",
            "units": "K",
        },
        "U": {
            "standard_name": "grid_eastward_wind",
            "long_name": "U-component of wind",
            "units": "m s-1",
        },
        "V": {
            "standard_name": "grid_northward_wind",
            "long_name": "V-component of wind",
            "units": "m s-1",
        },
        "W": {
            "standard_name": "upward_air_velocity",
            "long_name": "vertical wind velocity",
            "units": "m s-1",
        },
        "POT_VORTIC": {
            "standard_name": "ertel_potential_vorticity",
            "long_name": "potential vorticity",
            "units": "K m2 kg-1 s-1",
        },
    }

    def __init__(
        self,
        *,
        rotate_back: bool = False,
        pole_lon: float = 180.0,
        pole_lat: float = 90.0,
    ) -> None:
        """Create a new instance."""
        self.rotate_back: bool = rotate_back
        self.pole_lon: float = pole_lon
        self.pole_lat: float = pole_lat
        self._ds: xr.Dataset

    def convert(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert LAGRANTO dataset to COSMO format."""
        self._ds = ds
        ds_out = xr.Dataset(
            coords=self._convert_coords(),
            data_vars=self._convert_data_vars(),
            attrs=self._convert_attrs(),
        )
        del self._ds
        return ds_out

    def _convert_coords(self) -> dict[str, xr.DataArray]:
        """Convert coordinate arrays.

        Note: The LAGRANTO output format does not contain any coords, only
        regular data variables.

        """
        time_arr = self._prep_time_data("timedelta64[ns]")
        time_name = "time"
        da_time = xr.DataArray(
            name=time_name,
            data=time_arr,
            coords={time_name: time_arr},
            dims=(time_name,),
            attrs=dict(self.var_attrs_d.get(time_name, {})),
        )
        return {"time": da_time}

    def _convert_data_vars(self) -> dict[str, xr.DataArray]:
        """Convert data variables."""
        return {}

    def _convert_attrs(self) -> dict[str, Any]:
        """Convert global dataset attributes."""
        attrs: dict[str, Any] = {}
        shared = ["ref_year", "ref_month", "ref_day", "ref_hour", "ref_min", "ref_sec"]
        for name in shared:
            attrs[name] = self._ds.attrs.get(name, 0)
        attrs["duration_in_sec"] = (
            self._ds.attrs["duration"] * 60 if "duration" in self._ds.attrs else "n/a"
        )
        # -> Assumption: attrs_in['duration'] is always in minutes
        #    (which, according to Lukas J., may not always be the case)
        attrs["pollon"] = self.pole_lon
        attrs["pollat"] = self.pole_lat
        attrs["output_timestep_in_sec"] = self._comp_output_timestep_in_sec()
        attrs["_orig"] = dict(self._ds.attrs)
        return attrs

    def _comp_output_timestep_in_sec(self) -> int:
        """Compute output time step in seconds from time array."""
        time_ = self._prep_time_data("timedelta64[s]")
        steps: npt.NDArray[np.int_] = np.unique(time_[1:] - time_[:-1]).astype(int)
        if steps.size != 1:
            raise Exception(
                f"cannot derive time step: multiple different steps {steps} in {time_}"
            )
        # mypy 0.941 thinks result is of type Any (numpy 1.22.3)
        return cast(int, steps[0])

    def _prep_time_data(
        self, dtype: npt.DTypeLike = "timedelta64[s]"
    ) -> npt.NDArray[np.timedelta64]:
        """Prepare time coordinate data.

        The expected format in the input dataset is floats, whereby the integer
        components corresponds to hours, while the dicits after the point times
        100 represent fractional minutes.

        Example: `3.1533` is `3 h 15 min 20 s`.

        """
        time_in = self._ds.time.data[:, 0]
        abs_time_in = np.abs(time_in)
        time_h = abs_time_in.astype(np.int32)
        time_m = np.round((abs_time_in - time_h) * 100).astype(np.int32)
        time_s = ((((abs_time_in - time_h) * 100) - time_m) * 60).astype(np.int32)
        time_tot_s = (time_s + time_m * 60 + time_h * 60 * 60) * np.sign(time_in)
        return np.array(time_tot_s, "timedelta64[s]").astype(dtype)
