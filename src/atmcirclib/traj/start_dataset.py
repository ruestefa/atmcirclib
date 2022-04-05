"""Dataset with trajectories start points."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.typing import PathLike_T

if TYPE_CHECKING:
    # Local
    from .dataset import TrajDataset


class TrajStartPointDataset:
    """Trajectories start points."""

    def __init__(
        self,
        ds: xr.Dataset,
    ) -> None:
        """Create a new instance."""
        self.ds: xr.Dataset = ds

    def derive_bin_edges(
        self,
        *,
        nmax: Optional[Union[int, tuple[int, int, int]]] = None,
        extrapolate: bool = True,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Derive edges of point-centered bins in ``(lon, lat, z)`` coordinates.

        They correspond to the mid-points between successive points, plus an
        optional end point obtained by extrapolation.

        Args:
            nmax (optional): Maximum number of bins in all or each direction.

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

        def reduce_precision(
            points: npt.NDArray[np.float_], nmax: int, decimals: int = 6
        ) -> npt.NDArray[np.float_]:
            """Reduce the precision and thus the number of unique points."""
            while points.size > np.abs(nmax):
                points = np.unique(np.round(points, decimals=decimals))
                decimals -= 1
            return points

        def get_unique(name: str) -> npt.NDArray[np.float_]:
            """Get unique values of a variable."""
            orig = self.ds.variables[name].data
            uniq: npt.NDArray[np.float_] = np.unique(orig)
            if uniq.size < 2:
                raise Exception(
                    f"variable '{name}' (size {orig.size:,}) has too few unique points"
                    f" ({uniq.size} < 2): {uniq}"
                )
            return uniq

        centers_x: npt.NDArray[np.float_] = get_unique("longitude")
        centers_y: npt.NDArray[np.float_] = get_unique("latitude")
        centers_z: npt.NDArray[np.float_] = get_unique("z")

        if nmax is not None:
            if isinstance(nmax, int):
                nmax = (nmax, nmax, nmax)
            centers_x = reduce_precision(centers_x, nmax[0])
            centers_y = reduce_precision(centers_y, nmax[1])
            centers_z = reduce_precision(centers_z, nmax[2])

        edges_x = centers_to_edges(centers_x)
        edges_y = centers_to_edges(centers_y)
        edges_z = centers_to_edges(centers_z)

        return (edges_x, edges_y, edges_z)

    @classmethod
    def from_txt(
        cls,
        path: PathLike_T,
        *,
        n_header: int = 3,
        verbose: bool = False,  # TODO use proper logging
    ) -> TrajStartPointDataset:
        """Read start points from a text file."""

        def find_format(path: PathLike_T) -> str:
            """Find format of start file."""
            with open(path, "r") as f:
                head = [f.readline().strip() for _ in range(4)]
            if (
                head[0].startswith("Reference Date")
                and head[1] == "lon lat z"
                and head[2].startswith("----")
                and len(head[3].split()) == 3
            ):
                return "cosmo"
            elif all(len(line.split()) == 4 for line in head):
                return "lagra"
            else:
                raise Exception(
                    f"cannot determine format of start file '{path}':\n"
                    + "\n".join(head)
                )

        format_ = find_format(path)
        if verbose:
            print(f"read traj start points from '{format_}' start file '{path}'")
        if format_ == "cosmo":
            n_header = 3
            dtype = [("lon", "f4"), ("lat", "f4"), ("z", "f4")]
        elif format_ == "lagra":
            n_header = 0
            dtype = [("_", "f4"), ("lon", "f4"), ("lat", "f4"), ("z", "f4")]
        else:
            raise Exception(f"unknown start point format '{format_}'")
        arr: npt.NDArray[np.float_] = np.loadtxt(
            path,
            skiprows=n_header,
            dtype=dtype,
        )
        return cls(cls._init_dataset(arr["lon"], arr["lat"], arr["z"]))

    @classmethod
    def from_trajs(
        cls,
        trajs: TrajDataset,
        verbose: bool = False,  # TODO use proper logging
    ) -> TrajStartPointDataset:
        """Derive start points from trajectories."""
        if verbose:
            print("derive start points from trajs")
        idx = 0
        return cls(
            cls._init_dataset(
                trajs.get_data("longitude", idx_time=idx),
                trajs.get_data("latitude", idx_time=idx),
                trajs.get_data("z", idx_time=idx),
            )
        )

    @classmethod
    def from_txt_or_trajs(
        cls, path: Optional[PathLike_T], trajs: TrajDataset, *, verbose: bool = False
    ) -> TrajStartPointDataset:
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
