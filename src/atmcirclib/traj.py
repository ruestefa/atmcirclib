"""Work with trajectories."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import datetime as dt
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import Optional
from typing import overload
from typing import Union

# Third-party
import cartopy.crs as ccrs
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.geo import BoundingBox
from atmcirclib.typing import PathLike_T


class TrajsDataset:
    """A trajectories dataset as written by COSMO online trajs module."""

    @dc.dataclass
    class Config:
        """Configuration.

        Properties:
            nan: Missig value in data.

            boundary_size_km: Size of the domain boundary zone in kilometers.

            const_file: Path to file with constant model fields; required for
                removing trajs in the domain boundary zone etc.

            start_file: Path to text file with start points; required to provide
                start points, e.g., to a plotting routine to choose proper bins.

            start_file_header: Number of header lines in start file.

            verbose: Increase verbosity; TODO: replace by proper logging.

        """

        nan: float = -999.0
        boundary_size_km: float = 100.0
        const_file: Optional[PathLike_T] = None
        start_file: Optional[PathLike_T] = None
        start_file_header: int = 3
        verbose: bool = True

    class MissingConfigError(Exception):
        """Missing an entry in ``Config``."""

    def __init__(self, ds: xr.Dataset, **config_kwargs) -> None:
        """Create a new instance."""
        self.config: TrajsDataset.Config = self.Config(**config_kwargs)
        self.ds: xr.Dataset = ds

    def get_var_data(
        self,
        name: str,
        idx_time: Union[int, slice] = slice(None),
        idx_traj: Union[int, slice] = slice(None),
    ) -> npt.NDArray[np.float_]:
        """Get data (sub-) array of variable with NaNs as missing values."""
        arr = np.array(self.ds.variables[name].data[idx_time, idx_traj], np.float32)
        arr[arr == self.config.nan] = np.nan
        return arr

    def only(self, **criteria) -> TrajsDataset:
        """Return a copy with only those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(**criteria)
        return self._without_masked(~mask)

    def without(self, **criteria) -> TrajsDataset:
        """Return a copy without those trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        mask = self.get_traj_mask(**criteria)
        return self._without_masked(mask)

    def count(self, **criteria) -> int:
        """Count all trajs that fulfill the given criteria.

        See docstring of ``get_traj_mask`` for details on the criteria.

        """
        return self.get_traj_mask(**criteria).sum()

    def get_traj_mask(
        self,
        *,
        incomplete: Optional[bool] = None,
        boundary: Optional[bool] = None,
        uv: tuple[int, float, float] = None,
        z: tuple[int, float, float] = None,
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

    def open_const_file(self) -> xr.Dataset:
        """Open file ``Config.const_file``."""
        if self.config.const_file is None:
            raise self.MissingConfigError("const_file")
        return xr.open_dataset(self.config.const_file)

    def get_domain_proj(self) -> ccrs.Projection:
        """Get projection of simulation data."""
        with self.open_const_file() as ds:
            return ccrs.RotatedPole(
                pole_latitude=ds.rotated_pole.grid_north_pole_latitude,
                pole_longitude=ds.rotated_pole.grid_north_pole_longitude,
            )

    def get_bbox_xy(self, inner: bool = False) -> BoundingBox:
        """Get (lon, lat) bounding box, optionally minus boundary zone."""
        with self.open_const_file() as ds_const:
            bbox = BoundingBox.from_coords(ds_const.rlon.data, ds_const.rlat.data)
        if inner:
            km_per_deg = 110.0
            bbox = bbox.shrink(self.config.boundary_size_km / km_per_deg)
        return bbox

    def get_bbox_xz(self, inner: bool = False) -> BoundingBox:
        """Get (lon, z) bounding box, optionally minus horiz. boundary zone."""
        bbox_xy = self.get_bbox_xy(inner=inner)
        zmin = 0.0
        with self.open_const_file() as ds_const:
            try:
                zmax = float(ds_const.height_toa.data)
            except AttributeError:
                zmax = self.get_start_points(try_file=True)["z"].max()
        zmax /= 1000.0  # m => km
        return BoundingBox(
            llx=bbox_xy.llx,
            urx=bbox_xy.urx,
            lly=zmin,
            ury=zmax,
        )

    def get_bbox_yz(self, inner: bool = False) -> BoundingBox:
        """Get (lat, z) bounding box, optionally minus horiz. boundary zone."""
        bbox_xy = self.get_bbox_xy(inner=inner)
        bbox_xz = self.get_bbox_xz(inner=inner)
        return BoundingBox(
            llx=bbox_xy.lly,
            urx=bbox_xy.ury,
            lly=bbox_xz.lly,
            ury=bbox_xz.ury,
        )

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

        def pts_to_bins(pts: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
            """Obtain bin boundaries between points."""
            inner = np.mean([pts[:-1], pts[1:]], axis=0)
            if not extrapolate:
                return inner
            return np.r_[
                pts[0] - (inner[0] - pts[0]),
                inner,
                pts[-1] + (pts[-1] - inner[-1]),
            ]

        points = self.get_start_points(try_file=try_file)
        return (
            pts_to_bins(np.sort(np.unique(points["lon"]))),
            pts_to_bins(np.sort(np.unique(points["lat"]))),
            pts_to_bins(np.sort(np.unique(points["z"]))),
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

    @overload
    def get_abs_time(
        self,
        idcs: Union[Sequence[int], slice] = ...,
        *,
        format: Literal[False] = False,
    ) -> list[dt.datetime]:
        ...

    @overload
    def get_abs_time(
        self,
        idcs: Union[Sequence[int], slice] = ...,
        *,
        format: Literal[True],
    ) -> list[str]:
        ...

    def get_abs_time(self, idcs=slice(None), *, format=False):
        """Get the time dimension as absolute datimes.

        Args:
            idcs (optional): Indices or slice to return only a subset of steps.

            format (optional): Return formatted strings.

        """
        rel_time = self.ds.time.data[idcs].astype("timedelta64[s]").astype(dt.timedelta)
        abs_time = (self.get_dt_ref() + rel_time).tolist()
        if not format:
            return abs_time
        # return list(map(dt.datetime.strftime("%Y-%m-%d_%H:%M:%S"), abs_time))
        return [dt_.strftime("%Y-%m-%d %H:%M:%S") for dt_ in abs_time]

    @overload
    def get_duration(self, *, format: Literal[False] = False) -> dt.timedelta:
        ...

    @overload
    def get_duration(self, *, format: Literal[True]) -> str:
        ...

    def get_duration(self, *, format=False):
        """Get the duration of the dataset."""
        if format is False:
            return self.get_end() - self.get_start()
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

    @overload
    def get_start(self, *, format: Literal[False] = False) -> dt.datetime:
        ...

    @overload
    def get_start(self, *, format: Literal[True]) -> str:
        ...

    def get_start(self, *, format=False):
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
        return self.get_abs_time(idcs=[1], format=format)[0]

    @overload
    def get_end(self, *, format: Literal[False] = False) -> dt.datetime:
        ...

    @overload
    def get_end(self, *, format: Literal[True]) -> str:
        ...

    def get_end(self, *, format=False):
        """Get the last time step in the file, optionally as a string."""
        return self.get_abs_time(idcs=[-1], format=format)[0]

    def _without_masked(self, mask: npt.NDArray[np.bool_]) -> TrajsDataset:
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

    def _get_traj_mask_full(self, value: bool) -> npt.NDArray[np.bool_]:
        """Get a trajs mask."""
        return np.full(self.ds.dims["id"], value, np.bool_)

    def _get_traj_mask_any_incomplete(self) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs with any ``Config.nan`` values."""
        return (self.ds.z.data == self.config.nan).sum(axis=0) > 0

    def _get_traj_mask_any_boundary(self) -> npt.NDArray[np.bool_]:
        """Get 1D mask indicating all trajs ever reaching the boundary zone."""
        llrlon, urrlon, llrlat, urrlat = self.get_bbox_xy(inner=True)
        rlon, rlat = self.ds.longitude.data, self.ds.latitude.data
        nan_mask = self._get_traj_mask_any_incomplete()
        rlon_mask = np.where(nan_mask, True, (rlon < llrlon) | (rlon > urrlon))
        rlat_mask = np.where(nan_mask, True, (rlat < llrlat) | (rlat > urrlat))
        return (rlon_mask | rlat_mask).sum(axis=0) > 0

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

    @classmethod
    def from_file(cls, path: PathLike_T, **config_kwargs) -> TrajsDataset:
        """Read trajs dataset from file."""
        try:
            # ds = xr.open_dataset(path, engine="netcdf4")
            ds = xr.open_dataset(path)
        except Exception as e:
            raise ValueError(f"error opening trajectories files '{path}'") from e
        return cls(ds=ds, **config_kwargs)
