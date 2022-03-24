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
from atmcirclib.cosmo import COSMOGridDataset
from atmcirclib.typing import NDIndex_T
from atmcirclib.typing import PathLike_T


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
        arr: npt.NDArray[np.float32] = np.array(
            self.ds.variables[name].data[idx_time, idx_traj], np.float32
        )
        if replace_vnan:
            arr[arr == self.config.nan] = np.nan
        return arr

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

            boundary_size_deg: Size of the domain boundary zone in degrees.

            start_file: Path to text file with start points; required to provide
                start points, e.g., to a plotting routine to choose proper bins.

            start_file_header: Number of header lines in start file.

        """

        boundary_size_deg: float = 1.0  # TODO eliminate
        start_file: Optional[PathLike_T] = None
        start_file_header: int = 3

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        _grid: Optional[COSMOGridDataset] = None,  # TODO remove
        **config_kwargs: Any,
    ) -> None:
        """Create a new instance."""
        self.config: ExtendedTrajDataset.Config
        super().__init__(ds, **config_kwargs)
        self._grid: Optional[COSMOGridDataset] = _grid

    def only(self, **criteria: Any) -> TrajDataset:
        """Return a copy with only those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(**criteria)
        return self._without_masked(~mask)

    def without(self, **criteria: Any) -> TrajDataset:
        """Return a copy without those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(**criteria)
        return self._without_masked(mask)

    def count(self, **criteria: Any) -> int:
        """Count all trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(int, self.get_traj_mask(**criteria).sum())

    def get_traj_mask(
        self,
        *,
        incomplete: Optional[bool] = None,
        boundary: Optional[bool] = None,
        uv: Optional[tuple[int, float, float]] = None,
        z: Optional[tuple[int, float, float]] = None,
        require_all: bool = True,
    ) -> npt.NDArray[np.bool_]:
        """Get a mask indicating which trajs fulfill a combination of criteria.

        Args:
            incomplete (optional): Select trajs that are incomplete because they
                contain missing values, which is the case when they leave the
                domain; note that these are necessary a subset of those selected
                with ``boundary=True``.

            boundary (optional): Select trajs that enter the boundary zone
                (defined by ``Config.boundary_size_km``) at some point.

            uv (optional): Select trajs that at a given time step exhibit wind
                speed in a given range.

            z (optional): Select trajs that at a given time step are located in
                a given height range.

            require_all (optional): Only select trajs that fulfil all given
                criteria; otherwise, any one given criterion is sufficient.

        """

        def update_mask(
            mask: npt.NDArray[np.bool_], incr: npt.NDArray[np.bool_]
        ) -> None:
            """Update mask depending on ``require_all``."""
            if require_all:
                mask[:] &= incr
            else:
                mask[:] |= incr

        mask = self._get_traj_mask_full(require_all)
        if incomplete is not None:
            incr = self._get_traj_mask_any_incomplete()
            update_mask(mask, incr if incomplete else ~incr)
        if boundary is not None:
            incr = self._get_traj_mask_any_boundary()
            update_mask(mask, incr if boundary else ~incr)
        if uv is not None:
            update_mask(mask, self._get_traj_mask_uv(*uv))
        if z is not None:
            update_mask(mask, self._get_traj_mask_z(*z))
        return mask

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
        return type(self)(ds=new_ds, _grid=self._grid, **dc.asdict(self.config))

    def _get_traj_mask_full(self, value: bool) -> npt.NDArray[np.bool_]:
        """Get a trajs mask."""
        return np.full(self.ds.dims["id"], value, np.bool_)

    def _get_traj_mask_any_incomplete(self) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs with any ``Config.nan`` values."""
        mask = (self.ds.z.data == self.config.nan).sum(axis=0) > 0
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)

    def _get_traj_mask_any_boundary(self) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs ever reaching the boundary zone."""
        if self._grid is None:
            raise Exception("must pass _grid to identify boundary trajs")
        llrlon, urrlon, llrlat, urrlat = self._grid.get_bbox_xy().shrink(
            self.config.boundary_size_deg
        )
        rlon, rlat = self.ds.longitude.data, self.ds.latitude.data
        nan_mask = self._get_traj_mask_any_incomplete()
        rlon_mask = np.where(nan_mask, True, (rlon < llrlon) | (rlon > urrlon))
        rlat_mask = np.where(nan_mask, True, (rlat < llrlat) | (rlat > urrlat))
        mask = (rlon_mask | rlat_mask).sum(axis=0) > 0
        # mypy thinks return type is Any (mypy v0.941, numpy v1.22.3)
        return cast(npt.NDArray[np.bool_], mask)

    def _get_traj_mask_z(
        self, idx_time: int, vmin: Optional[float], vmax: Optional[float]
    ) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs that end in a given height range."""
        arr = self.ds.z.data[idx_time, :]
        return self._get_traj_mask_in_range(arr, vmin, vmax)

    def _get_traj_mask_uv(
        self, idx_time: int, vmin: Optional[float], vmax: Optional[float]
    ) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs that end in a given UV range."""
        arr = np.sqrt(
            self.ds.U.data[idx_time, :] ** 2 + self.ds.V.data[idx_time, :] ** 2
        )
        return self._get_traj_mask_in_range(arr, vmin, vmax)

    def _get_traj_mask_in_range(
        self, arr: npt.NDArray[np.float_], vmin: Optional[float], vmax: Optional[float]
    ) -> npt.NDArray[np.bool_]:
        """Get a task indicating trajs with a values of ``arr`` in range."""
        _name_ = "_get_traj_mask_in_range"
        if n_incomplete := self.count(incomplete=True):
            raise NotImplementedError(
                f"{type(self).__name__}.{_name_} for incomplete trajs"
                f" ({n_incomplete:,}/{self.count():,})"
            )
        mask = self._get_traj_mask_full(True)
        if vmin is not None:
            mask &= arr >= vmin
        if vmax is not None:
            mask &= arr <= vmax
        return mask
