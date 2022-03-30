"""Handle output files of LAGRANTO, like conversion to COSMO output format."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Any
from typing import cast

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# Local
from .dataset_ds import create_traj_dataset_ds

__all__: list[str] = [
    "convert_traj_ds_lagranto_to_cosmo",
]


def convert_traj_ds_lagranto_to_cosmo(ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
    """Convert a LAGRANTO output dataset to COSMO output format."""
    return TrajDsLagrantoToCosmoConverter(**kwargs).convert(ds)


@dc.dataclass
class TrajDsLagrantoToCosmoConverter:
    """Convert a LAGRANTO output dataset to COSMO output format."""

    rotate_back: bool = False
    pole_lon: float = 180.0
    pole_lat: float = 90.0

    def convert(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert LAGRANTO dataset to COSMO format."""
        dims: tuple[str, str] = ("", "")
        coords_d: dict[str, npt.NDArray[np.generic]] = {}
        data_d: dict[str, npt.NDArray[np.generic]] = {}
        dtype_d: dict[str, npt.DTypeLike] = {}
        scale_fact_d: dict[str, float] = {}
        attrs_d: dict[str, dict[str, Any]] = {}
        vnan: float = -999.0
        return create_traj_dataset_ds(
            dims=dims,
            coords_d=coords_d,
            data_d=data_d,
            dtype_d=dtype_d,
            scale_fact_d=scale_fact_d,
            attrs_d=attrs_d,
            attrs=self._convert_attrs(ds),
            vnan=vnan,
        )

    def _convert_attrs(self, ds: xr.Dataset) -> dict[str, Any]:
        """Convert global dataset attributes."""
        attrs: dict[str, Any] = {}
        shared = ["ref_year", "ref_month", "ref_day", "ref_hour", "ref_min", "ref_sec"]
        for name in shared:
            attrs[name] = ds.attrs.get(name, 0)
        attrs["duration_in_sec"] = (
            ds.attrs["duration"] * 60 if "duration" in ds.attrs else "n/a"
        )
        # -> Assumption: attrs_in['duration'] is always in minutes
        #    (which, according to Lukas J., may not always be the case)
        attrs["pollon"] = self.pole_lon
        attrs["pollat"] = self.pole_lat
        attrs["output_timestep_in_sec"] = self._comp_output_timestep_in_sec(ds)
        attrs["_orig"] = dict(ds.attrs)
        return attrs

    def _comp_output_timestep_in_sec(self, ds: xr.Dataset) -> int:
        """Compute output time step in seconds from time array."""
        time_ = self._prep_time_data(ds, "timedelta64[s]")
        steps: npt.NDArray[np.int_] = np.unique(time_[1:] - time_[:-1]).astype(int)
        if steps.size != 1:
            raise Exception(
                f"cannot derive time step: multiple different steps {steps} in {time_}"
            )
        # mypy 0.941 thinks result is of type Any (numpy 1.22.3)
        return cast(int, steps[0])

    @staticmethod
    def _prep_time_data(
        ds: xr.Dataset, dtype: npt.DTypeLike = "timedelta64[s]"
    ) -> npt.NDArray[np.timedelta64]:
        """Prepare time coordinate data.

        The expected format in the input dataset is floats, whereby the integer
        components corresponds to hours, while the dicits after the point times
        100 represent fractional minutes.

        Example: `3.1533` is `3 h 15 min 20 s`.

        """
        time_in = ds.time.data[:, 0]
        abs_time_in = np.abs(time_in)
        time_h = abs_time_in.astype(np.int32)
        time_m = np.round((abs_time_in - time_h) * 100).astype(np.int32)
        time_s = ((((abs_time_in - time_h) * 100) - time_m) * 60).astype(np.int32)
        time_tot_s = (time_s + time_m * 60 + time_h * 60 * 60) * np.sign(time_in)
        return np.array(time_tot_s, "timedelta64[s]").astype(dtype)
