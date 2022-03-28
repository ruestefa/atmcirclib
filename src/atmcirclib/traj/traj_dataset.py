"""Work with trajectories."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import datetime as dt
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import Optional
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.typing import NDIndex_T
from atmcirclib.typing import PathLike_T

# Local
from .criteria import Criteria
from .criteria import Criterion


class TrajDataset:
    """A trajectories dataset as written by COSMO online trajs module."""

    @dc.dataclass
    class Config:
        """Configuration.

        Properties:
            nan: Missig value in data.

            verbose: Increase verbosity; TODO: replace by proper logging.

        """

        nan: float = -999.0
        verbose: bool = True

    class MissingConfigError(Exception):
        """Missing an entry in ``Config``."""

    def __init__(self, ds: xr.Dataset, **config_kwargs: Any) -> None:
        """Create a new instance."""
        self.config: TrajDataset.Config = self.Config(**config_kwargs)
        self.ds: xr.Dataset = ds

    def count(self, criteria: Optional[Criteria] = None) -> int:
        """Count all trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(int, self.get_traj_mask(criteria).sum())

    def select(self, criteria: Optional[Criteria] = None) -> TrajDataset:
        """Return a copy with only those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(criteria)
        return self._without_masked(~mask)

    def get_traj_mask(
        self,
        criteria: Optional[Criteria] = None,
    ) -> npt.NDArray[np.bool_]:
        """Get a mask indicating which trajs fulfill a combination of criteria."""

        def update_mask(
            mask: npt.NDArray[np.bool_], criterion: Criterion, require_all: bool
        ) -> None:
            """Update mask depending on ``require_all``."""
            incr = criterion.apply(self)
            if require_all:
                mask[:] &= incr
            else:
                mask[:] |= incr

        if criteria is None:
            mask = Criterion.get_mask_full(self, value=True)
        else:
            mask = Criterion.get_mask_full(self, value=criteria.require_all)
            for criterion in criteria:
                update_mask(mask, criterion, criteria.require_all)
        return mask

    def get_data(
        self,
        name: str,
        idx_time: NDIndex_T = slice(None),
        idx_traj: NDIndex_T = slice(None),
        replace_vnan: bool = True,
    ) -> npt.NDArray[np.float_]:
        """Get data (sub-) array of variable with NaNs as missing values."""
        if idx_time is None:
            raise ValueError("idx_time must not be None; consider slice(None)")
        if idx_traj is None:
            raise ValueError("idx_traj must not be None; consider slice(None)")
        arr: npt.NDArray[np.float32]
        try:
            var = self.ds.variables[name]
        except KeyError as e:
            if name == "UV":
                # mypy 0.941 thinks result has dtype Any (numpy 1.22.3)
                return cast(
                    npt.NDArray[np.float_],
                    np.sqrt(
                        self.get_data("U", idx_time, idx_traj, replace_vnan) ** 2
                        + self.get_data("V", idx_time, idx_traj, replace_vnan) ** 2
                    ),
                )
            else:
                ds_vars = ", ".join(map("'{}'".format, self.ds.variables))
                dr_vars = ", ".join(map("'{}'".format, ["UV"]))
                raise ValueError(
                    f"invalid name '{name}'; neither in dataset ({ds_vars})"
                    f" nor among implemented derived variables ({dr_vars})"
                ) from e
        else:
            arr = np.array(var.data[idx_time, idx_traj], np.float32)
        if replace_vnan:
            arr[arr == self.config.nan] = np.nan
        return arr

    def _without_masked(self, mask: npt.NDArray[np.bool_]) -> TrajDataset:
        """Return a copy without those trajs indicated in the mask."""
        new_data_vars: dict[str, xr.DataArray] = {}
        for name, var in self.ds.data_vars.items():
            assert var.dims == ("time", "id")
            new_data_vars[name] = xr.DataArray(
                data=np.delete(var.data, mask, axis=1),
                coords=var.coords,
                dims=var.dims,
                name=var.name,
                attrs=dict(var.attrs),
                indexes=dict(var.indexes),
            )
        new_ds = xr.Dataset(
            data_vars=new_data_vars, coords=self.ds.coords, attrs=self.ds.attrs
        )
        return type(self)(ds=new_ds, **dc.asdict(self.config))

    @classmethod
    def from_file(cls, path: PathLike_T, **config_kwargs: Any) -> TrajDataset:
        """Read trajs dataset from file."""
        try:
            # ds = xr.open_dataset(path, engine="netcdf4")
            ds = xr.open_dataset(path)
        except Exception as e:
            raise ValueError(f"error opening trajectories files '{path}'") from e
        return cls(ds=ds, **config_kwargs)


# pylint: disable=R0904  # too-many-public-methods (>20)
class ExtendedTrajDataset(TrajDataset):
    """A temporary extension of ``TrajDataset`` with additional methods.

    These methods are untested and might be moved to other classes eventually.
    Only the interface of ``TrajDataset`` can be considered stable!

    """

    @dc.dataclass
    class Config(TrajDataset.Config):
        """Configuration.

        Properties:

            start_file: Path to text file with start points; required to provide
                start points, e.g., to a plotting routine to choose proper bins.

            start_file_header: Number of header lines in start file.

        """

        start_file: Optional[PathLike_T] = None
        start_file_header: int = 3

    def __init__(
        self,
        ds: xr.Dataset,
        **config_kwargs: Any,
    ) -> None:
        """Create a new instance."""
        self.config: ExtendedTrajDataset.Config
        super().__init__(ds, **config_kwargs)

    def remove(self, criteria: Optional[Criteria] = None) -> TrajDataset:
        """Return a copy without those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(criteria)
        return self._without_masked(mask)

    def discount(self, criteria: Optional[Criteria] = None) -> int:
        """Count all trajs that don't fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        return self.count() - self.count(criteria)

    # Note: Typing return array as npt.NDArray[np.float_] leads to overload error
    # when accessing fields by name (e.g., points["x"]) (numpy v1.22.2)
    def get_start_points(self, try_file: bool = False) -> npt.NDArray[Any]:
        """Get start points from trajs as a structured array.

        Args:
            try_file (optional): Try first to read start points from file.

        """
        if try_file:
            try:
                return self.read_start_points()
            except (self.MissingConfigError, FileNotFoundError) as e:
                if self.config.verbose:
                    print(f"reading trajs from file filed ({type(e).__name__})")
        if self.config.verbose:
            print("get start points from trajs")
        points = np.rec.fromarrays(
            [self.ds.longitude.data[0], self.ds.latitude.data[0], self.ds.z.data[0]],
            dtype=[("lon", "f4"), ("lat", "f4"), ("z", "f4")],
        )
        return np.unique(points)

    # Note: Typing return array as npt.NDArray[np.float_] leads to overload error
    # when accessing fields by name (e.g., points["x"]) (numpy v1.22.2)
    def read_start_points(self) -> npt.NDArray[Any]:
        """Read start points from ``Config.start_file`` as a structured array."""
        if self.config.start_file is None:
            raise self.MissingConfigError("start_file")
        if self.config.verbose:
            print(f"read traj start points from {self.config.start_file}")
        return np.loadtxt(
            self.config.start_file,
            skiprows=self.config.start_file_header,
            dtype=[("lon", "f4"), ("lat", "f4"), ("z", "f4")],
        )

    def get_start_point_bin_edges(
        self, *, try_file: bool = True, extrapolate: bool = True
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Compute bins in (lon, lat, z) from trajs start points.

        They correspond to the mid-points between successive points, plus an
        optional end point obtained by extrapolation.

        Args:
            try_file (optional): Try first to read start points from file.

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

        points = self.get_start_points(try_file=try_file)
        return (
            centers_to_edges(np.sort(np.unique(points["lon"]))),
            centers_to_edges(np.sort(np.unique(points["lat"]))),
            centers_to_edges(np.sort(np.unique(points["z"]))),
        )

    def get_dt_ref(self) -> dt.datetime:
        """Get reference datetime (start of the simulation)."""
        return dt.datetime(
            self.ds.attrs["ref_year"],
            self.ds.attrs["ref_month"],
            self.ds.attrs["ref_day"],
            self.ds.attrs["ref_hour"],
            self.ds.attrs["ref_min"],
            self.ds.attrs["ref_sec"],
        )

    def get_abs_time(
        self, idcs: Union[Sequence[int], slice] = slice(None)
    ) -> list[dt.datetime]:
        """Get the time dimension as absolute datimes.

        Args:
            idcs (optional): Indices or slice to return only a subset of steps.

        """
        rel_time = self.ds.time.data[idcs].astype("timedelta64[s]").astype(dt.timedelta)
        abs_time = (self.get_dt_ref() + rel_time).tolist()
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(list[dt.datetime], abs_time)

    def format_abs_time(
        self,
        idcs: Union[Sequence[int], slice] = slice(None),
    ) -> list[str]:
        """Format the time dimension as absolute datimes."""
        abs_time = self.get_abs_time(idcs)
        return [dt_.strftime("%Y-%m-%d %H:%M:%S") for dt_ in abs_time]

    def get_duration(self) -> dt.timedelta:
        """Get the duration of the dataset."""
        return self.get_end() - self.get_start()

    def format_duration(self) -> str:
        """Format the duration of the dataset."""
        return self.format_rel_time(self.get_end())

    def format_rel_time(
        self,
        end: dt.datetime,
        start: Optional[dt.datetime] = None,
    ) -> str:
        """Format relative, by default since the start of the dataset."""
        if start is None:
            start = self.get_start()
        dur = end - start
        tot_secs = dur.total_seconds()
        hours = int(tot_secs / 3600)
        mins = int((tot_secs - 3600 * hours) / 60)
        secs = int(tot_secs - 3600 * hours - 60 * mins)
        assert secs + 60 * mins + 3600 * hours == tot_secs
        return f"{hours:02}:{mins:02}:{secs:02}"

    def get_start(self) -> dt.datetime:
        """Get the first time step in the file, optionally as a string."""
        # Note (2022-02-04):
        # Don't use the first time step because it corresponds to the last
        # model time step before the start of the trajs (the time always
        # corresponds to the end of the model time step AFTER the trajs have
        # been incremented)
        # Example for dt=10s, dt_trace=60s:
        #   [2016-09-24_23:59:50, 2016-09-25_00:00:00, 2016-09-25_00:01:00, ...]
        # What we want instead is the second step.
        # Eventually, this should be implemented in COSMO more consistently!
        return self.get_abs_time(idcs=[1])[0]

    def format_start(self) -> str:
        """Format the first time step in the file, optionally as a string."""
        # See comment in method get_start
        return self.format_abs_time(idcs=[1])[0]

    def get_end(self) -> dt.datetime:
        """Get the last time step in the file, optionally as a string."""
        return self.get_abs_time(idcs=[-1])[0]

    def format_end(self) -> str:
        """Format the last time step in the file, optionally as a string."""
        return self.format_abs_time(idcs=[-1])[0]
