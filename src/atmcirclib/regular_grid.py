"""Read and plot regular model grids."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import cast
from typing import Optional
from typing import TypeVar
from typing import Union

# Third-party
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import numpy.typing as npt
from cartopy.mpl.contour import GeoContourSet
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
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
        self.topo = topo
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
    def from_cosmo_file(
        cls, path: Path, *, n_bnd: int = 0, read_topo: bool = False
    ) -> RegularGrid:
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
            if read_topo:
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
        *,
        levels: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
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
        *,
        levels: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
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


@dc.dataclass
class FontSizes:
    """Large, medium and small font sizes."""

    l: float = 14
    m: float = 12
    s: float = 10

    def scale(self, f: float) -> FontSizes:
        """Return scaled font sizes."""
        return type(self)(
            l=self.l * f,  # noqa: disable=E741  # ambiguous variable name 'l'
            m=self.m * f,
            s=self.s * f,
        )


class RegularGridPlot:
    """Plot one or more regular model grids with topography etc."""

    def __init__(
        self,
        grid_or_grids: Union[RegularGrid, Sequence[RegularGrid]],
        *,
        scale: float = 1.0,
    ) -> None:
        """Create a new instance."""
        if not isinstance(grid_or_grids, Sequence):
            grid_or_grids = [grid_or_grids]
        if len(grid_or_grids) < 1:
            raise ValueError("must pass at least one grid")
        self.grids: list[RegularGrid] = list(grid_or_grids)
        self.main_grid: RegularGrid = next(iter(self.grids))
        self.scale = scale

        self.fs = FontSizes().scale(self.scale)

        self.fig: Figure = plt.figure(figsize=(8 * self.scale, 8 * self.scale))
        self.fig.set_facecolor("white")

        self.ax: Axes = self.fig.add_axes(
            [0.05, 0.05, 0.9, 0.9], projection=self.main_grid.get_proj()
        )
        self.ax.set_adjustable("box")
        self.ax.set_extent(
            self.main_grid.get_extent(grow=0.03), crs=self.main_grid.get_proj()
        )

        self._grid_pltrs = [RegularGridPlotter(grid) for grid in self.grids]
        self._main_grid_pltr = next(iter(self._grid_pltrs))

        self._topo_con_handles: list[GeoContourSet] = []
        self._topo_col_handles: list[GeoContourSet] = []
        self._outline_labels: list[Optional[str]] = []
        self._outline_handles: list[Line2D] = []

    def add_grid_lines(self) -> None:
        """Add grid lines."""
        self._main_grid_pltr.add_grid_lines(
            self.ax,
            linewidth=1.0 * self.scale,
            fontsize=self.fs.m,
        )

    def add_outlines(
        self,
        *,
        labels: Optional[Sequence[Optional[str]]] = None,
        linewidths: Optional[
            Union[Sequence[Optional[float]], npt.NDArray[np.float_]]
        ] = None,
        linestyles: Optional[Sequence[Optional[str]]] = None,
        scale_linewidths: bool = True,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Add outlines of all grids.

        Args:
            labels (optional): Label for each grid; omit individuals with None.

            linewidths (optional): Line width for each grid; omit individual
                ones with None.

            linestyles (optional): Line style for each grid; omit individual ones
                with None.

            scale_linewidths (optional): Multiply line widths with
                ``self.scale``.

            **kwargs: Additional arguments passed on to ``self.add_outline``.

        """
        T = TypeVar("T")

        def init_check_len(
            seq: Optional[Sequence[Optional[T]]], name: str
        ) -> list[Optional[T]]:
            """Initialize optional sequence if None or check length otherwise."""
            n = len(self.grids)
            if seq is None:
                return [None] * n
            if (ni := len(seq)) != n:
                raise ValueError(f"wrong number of {name}: expected {n}, got {ni}")
            return list(seq)

        labels = init_check_len(labels, "labels")
        if isinstance(linewidths, np.ndarray):
            linewidths = cast(list[float], linewidths.tolist())
        linewidths = init_check_len(linewidths, "linewidths")
        if scale_linewidths:
            linewidths = [lw * self.scale if lw else lw for lw in linewidths]
        linestyles = init_check_len(linestyles, "linestyles")

        handles: list[Line2D] = []
        for grid, label, lw, ls in zip(self.grids, labels, linewidths, linestyles):
            kwargs_i = {"linewidth": lw, "linestyle": ls, **kwargs}
            handles.append(self.add_outline(grid=grid, label=label, **kwargs_i))
        return handles

    def add_outline(
        self,
        *,
        grid: Optional[RegularGrid] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Line2D:
        """Add grid outline, by default of the main grid."""
        pltr = self._get_pltr(grid)
        handle = pltr.add_outline(self.ax, **kwargs)
        self._outline_handles.append(handle)
        self._outline_labels.append(label)
        return handle

    def add_outline_legend(self, **kwargs: Any) -> None:
        """Add legend for domain outlines."""
        y0 = self._get_legend_y0()
        bbox_to_anchor_ = (0.075, y0, 0.85, -y0)
        kwargs = {
            "loc": "lower center",
            "bbox_to_anchor": bbox_to_anchor_,
            "frameon": False,
            "mode": "expand",
            "ncol": 2,
            "fontsize": self.fs.l,
            "handlelength": 2.5,
            **kwargs,
        }
        self.fig.legend(
            handles=self._outline_handles,
            labels=self._outline_labels,
            **kwargs,
        )

    def add_topos(
        self,
        *,
        levels_col: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
        levels_con: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
    ) -> None:
        """Add topography of all grids."""
        for grid in self.grids:
            if grid.topo is not None:
                self.add_topo(grid=grid, levels_col=levels_col, levels_con=levels_con)

    def add_topo(
        self,
        *,
        levels_col: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
        levels_con: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
        plot_col: bool = True,
        plot_con: bool = True,
        grid: Optional[RegularGrid] = None,
    ) -> None:
        """Add topography, by default of the main grid."""
        pltr = self._get_pltr(grid)
        if plot_col:
            handle = pltr.add_topo_colors(self.ax, levels=levels_col)
            self._topo_col_handles.append(handle)
        if plot_con:
            handle = pltr.add_topo_contours(
                self.ax, levels=levels_con, linewidths=0.8 * self.scale
            )
            self._topo_con_handles.append(handle)

    def add_topo_cbar(
        self,
        *,
        label: str = "Model surface height (m)",
        ticks: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None,
        bbox_to_anchor: Optional[tuple[float, float, float, float]] = None,
    ) -> Colorbar:
        """Add topography color bar using the latest color plot handle.

        Note that because the default vertical position depends on the size of
        the outline legend box between the plot and the color bar, the outlines
        need to be added to the plot before the color bar.

        """
        try:
            col_handle = self._topo_col_handles[-1]
        except IndexError as e:
            raise Exception("must add topo color field before adding cbar") from e
        if bbox_to_anchor is None:
            y0 = self._get_cbar_y0()
            bbox_to_anchor = (0.05, y0, 0.9, 0.04)
        ax_cb: Axes = self.fig.add_axes(bbox_to_anchor)
        cb: Colorbar = self.fig.colorbar(
            col_handle, cax=ax_cb, ticks=ticks, orientation="horizontal"
        )
        try:
            con_handle = self._topo_con_handles[-1]
        except IndexError:
            pass
        else:
            cb.add_lines(con_handle)
        cb.ax.tick_params(labelsize=self.fs.m)
        cb.set_label(label, fontsize=self.fs.l)
        return cb

    def _get_pltr(self, grid: Optional[RegularGrid]) -> RegularGridPlotter:
        """Get plotter of default or custom grid."""
        if grid is None:
            return self._main_grid_pltr
        return RegularGridPlotter(grid)

    def _get_legend_y0(self) -> float:
        """Get vertical baseline position for legend box below plot."""
        nl = len(self._outline_labels)
        # Empirical formula based on one and two rows of labels
        # May need to be adjusted for more rows (not tested)
        return -0.03 - (int(nl / 2 + 0.5) - 1) * 0.07

    def _get_cbar_y0(self) -> float:
        """Get vertical baseline position of color bar below plot and legend."""
        # Empirical formula based on up to two rows of legend
        return self._get_legend_y0() - 0.05
