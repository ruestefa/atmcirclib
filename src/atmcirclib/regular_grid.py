"""Read and plot regular model grids."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import cast
from typing import Optional
from typing import Union

# Third-party
import cartopy.crs as ccrs
import matplotlib as mpl
import netCDF4 as nc4
import numpy as np
import numpy.typing as npt
from cartopy.mpl.contour import GeoContourSet
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D

# Local
from .plot_utils import concat_cmaps
from .plot_utils import trunc_cmap


class RegularGrid:
    """Lat/lon grid with rotated pole."""

    def __init__(
        self,
        lat1d: npt.NDArray[np.float_],
        lon1d: npt.NDArray[np.float_],
        *,
        pole_lat: float = 90.0,
        pole_lon: float = 180.0,
        topo: Optional[npt.NDArray[np.float_]] = None,
    ) -> None:
        """Create a new instance."""
        self.lat1d = lat1d
        self.lon1d = lon1d
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
        self.topo: npt.NDArray[np.float_] = (
            np.zeros(self.get_shape(), np.float32) if topo is None else topo
        )
        self.ll_lat: float = self.lat1d[0]
        self.ll_lon: float = self.lon1d[0]
        self.ur_lat: float = self.lat1d[-1]
        self.ur_lon: float = self.lon1d[-1]
        self.c_lat: float = cast(float, np.mean([self.ll_lat, self.ur_lat]))
        self.c_lon: float = cast(float, np.mean([self.ll_lon, self.ur_lon]))
        self.lat_extent: float = self.ur_lat - self.ll_lat
        self.lon_extent: float = self.ur_lon - self.ll_lon
        self.outline: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]] = (
            np.array([self.ll_lon, self.ur_lon, self.ur_lon, self.ll_lon, self.ll_lon]),
            np.array([self.ll_lat, self.ll_lat, self.ur_lat, self.ur_lat, self.ll_lat]),
        )

    def get_shape(self) -> tuple[int, int]:
        """Get shape of (lat, lon) field."""
        return (self.lat1d.size, self.lon1d.size)

    def get_proj(self) -> ccrs.Projection:
        """Return the grid projection."""
        if not np.allclose((self.pole_lat, self.pole_lon), (90.0, 180.0)):
            return ccrs.RotatedPole(
                pole_latitude=self.pole_lat, pole_longitude=self.pole_lon
            )
        return ccrs.PlateCarree()

    def get_extent(self, *, grow: float = 0.0) -> tuple[float, float, float, float]:
        """Get domain extent, optionally enlarged by relative factor ``grow``."""
        return (
            self.ll_lon - 0.5 * grow * self.lon_extent,
            self.ur_lon + 0.5 * grow * self.lon_extent,
            self.ll_lat - 0.5 * grow * self.lat_extent,
            self.ur_lat + 0.5 * grow * self.lat_extent,
        )

    def get_bnd_w(self) -> npt.NDArray[np.float_]:
        """Compute geographic lat/lon coordinates of western boundary."""
        proj_geo = ccrs.PlateCarree()
        lons, lats, _ = proj_geo.transform_points(
            self.get_proj(), np.array([self.ll_lon] * self.lat1d.size), self.lat1d
        ).T
        return np.array([lats, lons])

    def get_bnd_s(self) -> npt.NDArray[np.float_]:
        """Compute geographic lat/lon coordinates of southern boundary."""
        proj_geo = ccrs.PlateCarree()
        lons, lats, _ = proj_geo.transform_points(
            self.get_proj(), self.lon1d, np.array([self.ll_lat] * self.lon1d.size)
        ).T
        return np.array([lats, lons])

    def get_bnd_e(self) -> npt.NDArray[np.float_]:
        """Compute geographic lat/lon coordinates of eastern boundary."""
        proj_geo = ccrs.PlateCarree()
        lons, lats, _ = proj_geo.transform_points(
            self.get_proj(), np.array([self.ur_lon] * self.lat1d.size), self.lat1d
        ).T
        return np.array([lats, lons])

    def get_bnd_n(self) -> npt.NDArray[np.float_]:
        """Compute geographic lat/lon coordinates of northern boundary."""
        proj_geo = ccrs.PlateCarree()
        lons, lats, _ = proj_geo.transform_points(
            self.get_proj(), self.lon1d, np.array([self.ur_lat] * self.lon1d.size)
        ).T
        return np.array([lats, lons])

    @classmethod
    def from_cosmo_file(cls, path: Path, n_bnd: int = 0) -> RegularGrid:
        """Read grid information and (optionally) grid arrays from file."""
        print(f"{cls.__name__}.from_cosmo_file({path}, {n_bnd=})")
        kwargs = {}
        idcs = slice(None) if not n_bnd else slice(n_bnd, -n_bnd)
        with nc4.Dataset(path, "r") as fi:
            kwargs["lat1d"] = fi.variables["rlat"][idcs]
            kwargs["lon1d"] = fi.variables["rlon"][idcs]
            pole_var = fi.variables["rotated_pole"]
            kwargs["pole_lat"] = pole_var.getncattr("grid_north_pole_latitude")
            kwargs["pole_lon"] = pole_var.getncattr("grid_north_pole_longitude")
            topo = fi.variables["HSURF"][0, idcs, idcs]
            land = fi.variables["FR_LAND"][0, idcs, idcs]
            kwargs["topo"] = np.where((topo > 0) & (land > 0), topo, 0.0)
        return cls(**kwargs)


@dc.dataclass
class RegularGridPlotter:
    """Plot elements of a rotated grid on a map plot axes."""

    grid: RegularGrid

    def add_outline(self, ax: Axes, **kwargs: Any) -> Line2D:
        """Plot grid outline onto an axes."""
        kwargs = {
            "color": "black",
            "zorder": self.get_zorder("grid_outlines"),
            **kwargs,
        }
        lines = ax.plot(
            *self.grid.outline,
            transform=self.grid.get_proj(),
            **kwargs,
        )
        assert len(lines) == 1
        return next(iter(lines))

    def add_grid_lines(
        self,
        ax: GeoAxes,
        *,
        d_lat: float = 10,
        d_lon: float = 10,
        fontsize: Union[str, float] = "medium",
        **kwargs: Any,
    ) -> None:
        """Add labeled lat/lon grid lines to a map plot axes."""

        def is_in_range(val: float, arr: npt.NDArray[np.float_]) -> bool:
            """Check if a number is in the range of an array."""
            return cast(bool, np.nanmin(arr) <= val <= np.nanmax(arr))

        lons = np.arange(-180, 180.1, d_lon)
        lats = np.arange(-90, 90.1, d_lat)

        kwargs = {
            "color": "black",
            "linewidth": 1.0,
            "alpha": 0.5,
            "zorder": self.get_zorder("gridlines"),
            **kwargs,
        }

        # Draw grid lines
        gl = ax.gridlines(draw_labels=False, **kwargs)
        gl.xlocator = mpl.ticker.FixedLocator(lons)
        gl.ylocator = mpl.ticker.FixedLocator(lats)

        # Determine labels to show on each side of the plot
        labels_left = [v for v in lats if is_in_range(v, self.grid.get_bnd_w()[0])]
        labels_bottom = [v for v in lons if is_in_range(v, self.grid.get_bnd_s()[1])]
        labels_right = [v for v in lats if is_in_range(v, self.grid.get_bnd_e()[0])]
        labels_top = [v for v in lons if is_in_range(v, self.grid.get_bnd_n()[1])]

        gl_kwargs = {
            "draw_labels": True,
            "x_inline": False,
            "y_inline": False,
            "zorder": kwargs["zorder"],
            "alpha": 0.0,
        }

        # Add labels to bottom and right-hand sides of plot
        gl = ax.gridlines(**gl_kwargs)
        gl.xlocator = mpl.ticker.FixedLocator(labels_bottom)
        gl.ylocator = mpl.ticker.FixedLocator(labels_right)
        gl.rotate_labels = False
        gl.left_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size": fontsize}
        gl.ylabel_style = {"size": fontsize}

        # Add labels to top and left-hand sides of plot
        gl = ax.gridlines(**gl_kwargs)
        gl.xlocator = mpl.ticker.FixedLocator(labels_top)
        gl.ylocator = mpl.ticker.FixedLocator(labels_left)
        gl.rotate_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.xlabel_style = {"size": fontsize}
        gl.ylabel_style = {"size": fontsize}

    def add_topo_colors(
        self,
        ax: GeoAxes,
        levels: Union[Sequence[float], npt.NDArray[np.float_]],
        **kwargs: Any,
    ) -> GeoContourSet:
        """Plot model topography on a map plot axes."""
        kwargs = {
            "cmap": self.get_cmap(),
            "zorder": self.get_zorder("topography"),
            "extend": "both",
            **kwargs,
        }
        return ax.contourf(
            self.grid.lon1d,
            self.grid.lat1d,
            self.grid.topo,
            transform=self.grid.get_proj(),
            levels=levels,
            **kwargs,
        )

    def add_topo_contours(
        self,
        ax: GeoAxes,
        levels: Union[Sequence[float], npt.NDArray[np.float_]],
        **kwargs: Any,
    ) -> GeoContourSet:
        """Plot model topography on a map plot axes."""
        kwargs = {
            "zorder": self.get_zorder("topography"),
            "colors": "black",
            "alpha": 0.75,
            "linewidths": 0.8,
            **kwargs,
        }
        return ax.contour(
            self.grid.lon1d,
            self.grid.lat1d,
            self.grid.topo,
            transform=self.grid.get_proj(),
            levels=levels,
            **kwargs,
        )

    def get_zorder(self, name: str) -> int:
        """Get zorder of elements by name."""
        return ["topography", "coastlines", "gridlines", "grid_outlines"].index(
            name
        ) + 10

    def get_cmap(self) -> Colormap:
        """Get default topographic color map."""
        cmap = concat_cmaps(
            trunc_cmap("PiYG_r", 0.0, 0.5),
            trunc_cmap("BrBG_r", 0.5, 1.0),
        )
        cmap.set_under("cornflowerblue")
        cmap.set_over("red")
        return cmap
