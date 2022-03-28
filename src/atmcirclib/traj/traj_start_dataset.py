"""Dataset with trajectories start points."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.typing import PathLike_T

if TYPE_CHECKING:
    # Local
    from .traj_dataset import TrajDataset


class TrajStartDataset:
    """Trajectories start points."""

    def __init__(
        self,
        ds: xr.Dataset,
    ) -> None:
        """Create a new instance."""
        self.ds: xr.Dataset = ds

    def derive_bin_edges(
        self, *, extrapolate: bool = True
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Derive edges of point-centered bins in ``(lon, lat, z)`` coordinates.

        They correspond to the mid-points between successive points, plus an
        optional end point obtained by extrapolation.

        Args:
            extrapolate (optional): Add additional bins in the beginning and
                end that include the first and last point, respectively, along
                each dimension.

        """

        def centers_to_edges(centers: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
            """Obtain bin boundaries between points."""
            inner_edges = np.mean([centers[:-1], centers[1:]], axis=0)
            if not extrapolate:
                edges = inner_edges
            else:
                edges = np.r_[
                    centers[0] - (inner_edges[0] - centers[0]),
                    inner_edges,
                    centers[-1] + (centers[-1] - inner_edges[-1]),
                ]
            # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
            return cast(npt.NDArray[np.float_], edges)

        return (
            centers_to_edges(np.unique(self.ds.longitude.data)),
            centers_to_edges(np.unique(self.ds.latitude.data)),
            centers_to_edges(np.unique(self.ds.z.data)),
        )

    @classmethod
    def from_txt(
        cls,
        path: PathLike_T,
        *,
        n_header: int = 3,
        verbose: bool = False,  # TODO use proper logging
    ) -> TrajStartDataset:
        """Read start points from a text file."""
        if verbose:
            print(f"read traj start points from {path}")
        arr: npt.NDArray[np.float_] = np.loadtxt(
            path,
            skiprows=n_header,
            dtype=[("lon", "f4"), ("lat", "f4"), ("z", "f4")],
        )
        return cls(cls._init_dataset(arr["lon"], arr["lat"], arr["z"]))

    @classmethod
    def from_trajs(
        cls,
        trajs: TrajDataset,
        verbose: bool = False,  # TODO use proper logging
    ) -> TrajStartDataset:
        """Derive start points from trajectories."""
        if verbose:
            print("derive start points from trajs")
        return cls(
            cls._init_dataset(
                trajs.ds.longitude.data[0],
                trajs.ds.latitude.data[0],
                trajs.ds.z.data[0],
            )
        )

    @classmethod
    def from_txt_or_trajs(
        cls, path: Optional[PathLike_T], trajs: TrajDataset, *, verbose: bool = False
    ) -> TrajStartDataset:
        """Try to read from file; if that fails, derive from trajectories."""
        if path is not None:
            try:
                return cls.from_txt(path, verbose=verbose)
            except IOError as e:
                if verbose:
                    print(
                        f"error reading start file '{path}'"
                        f": {type(e).__name__}('{str(e)}')"
                        "; derive start points from trajs instead"
                    )
        return cls.from_trajs(trajs, verbose=verbose)

    @staticmethod
    def _init_dataset(
        lon: npt.NDArray[np.float_],
        lat: npt.NDArray[np.float_],
        z: npt.NDArray[np.float_],
        attrs: Optional[dict[str, Any]] = None,
    ) -> xr.Dataset:
        """Create a dataset from a plain points array."""
        assert lon.size == lat.size == z.size  # TODO proper check
        idcs = np.arange(lon.size)
        coord_name = "i"
        coord = xr.DataArray(
            data=idcs,
            coords={coord_name: idcs},
            dims=(coord_name,),
            name=coord_name,
            attrs={
                "long_name": "start point index",
                "description": (
                    "caution: start point indices are NOT the same as the"
                    " trajectory ids in the model output"
                ),
            },
        )
        lon_name = "longitude"
        lon_var = xr.DataArray(
            data=lon,
            dims=(coord_name,),
            name=lon_name,
            attrs={
                "standard_name": "grid_longitude",
                "long_name": "rotated longitudes",
                "units": "degrees",
            },
        )
        lat_name = "latitude"
        lat_var = xr.DataArray(
            data=lat,
            dims=(coord_name,),
            name=lat_name,
            attrs={
                "standard_name": "grid_latitude",
                "long_name": "rotated latitudes",
                "units": "degrees",
            },
        )
        z_name = "z"
        z_var = xr.DataArray(
            data=z,
            dims=(coord_name,),
            name=z_name,
            attrs={
                "standard_name": "height",
                "long_name": "height above mean sea level",
                "units": "m AMSL",
            },
        )
        return xr.Dataset(
            coords={coord_name: coord},
            data_vars={lon_name: lon_var, lat_name: lat_var, z_name: z_var},
            attrs=attrs or {},
        )
