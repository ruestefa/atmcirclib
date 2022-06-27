"""Plot three fields like low-, mid- and high-level clouds together in RGB plot."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Any
from typing import cast
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits as mplt
import mpl_toolkits.axes_grid1
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np
import numpy.typing as npt

mpl.use("Agg")


# Derived types
ExtentT = Tuple[float, float, float, float]
# XYLimT = Optional[tuple[Optional[int], Optional[int]]]


def not_none(val: Optional[Any], default: Any) -> Any:
    """Return ``val`` unless it is None, in which case return ``default``."""
    if val is not None:
        return val
    return default


@dc.dataclass
class FontProperties:
    """Font properties like sizes."""

    large: int = 9
    medium: int = 8
    small: int = 7

    def bigger(self, n: int) -> FontProperties:
        """Increase font sizes by ``n`` points."""
        return FontProperties(
            large=self.large + n,
            medium=self.medium + n,
            small=self.small + n,
        )

    def smaller(self, n: int) -> FontProperties:
        """Decrease font sizes by ``n`` points."""
        return self.bigger(-n)


def format_file_name(time_idx: int) -> str:
    """Format a COSMO output file name by inserting the time step."""
    day = int(time_idx / 24)
    hour = time_idx % 24
    ddhh = f"{day:02d}{hour:02d}"
    return f"lfff{ddhh}0000.nc"


def km2deg(val: float) -> float:
    """Convert kilometers to degrees (approximate)."""
    return val / 110.0


def luminance(r: float, g: float, b: float) -> float:
    """Compute the luminance of an RBG (0..1) color.

    Source: https://stackoverflow.com/a/3943023/4419816

    """

    def adjust(c: float) -> float:
        if c <= 0.03928:
            return c / 12.92
        # mypy 0.941 thinks return value is of type Any
        return cast(float, ((c + 0.055) / 1.055) ** 2.4)

    return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)


def bw_contrast(r: float, g: float, b: float) -> str:
    """Return black or white depending on the background RBG (0..1).

    Source: https://stackoverflow.com/a/3943023/4419816

    """
    if luminance(r, g, b) > np.sqrt(1.05 * 0.05) - 0.05:
        return "black"
    return "white"


def logistic(x: float, x0: float, k: float = 1.0) -> float:
    """Compute the value of a logistic functions at a certain point."""
    # mypy 0.941 thinks np.exp returns Any (numpy 1.22.3)
    return cast(float, 1.0 / (1.0 + np.exp(-k * (x - x0))))


class RGBPlot:
    """A plot of a three-component field (like clouds) with RGB colors."""

    def __init__(
        self,
        arr: npt.NDArray[np.float_],
        data_extent: ExtentT,
        *,
        pollat: float = 90.0,
        pollon: float = 0.0,
        dpi: float = 300,
    ) -> None:
        """Create an instance of ``RGBPlot``."""
        self.arr: npt.NDArray[np.float_] = arr
        self.data_extent: ExtentT = data_extent
        # self.pollat: float = pollat
        # self.pollon: float = pollon

        self.proj_geo = ccrs.PlateCarree()
        self.proj_data = ccrs.RotatedPole(pole_latitude=pollat, pole_longitude=pollon)
        self.proj_map = self.proj_data

        self.font: FontProperties = FontProperties()

        self.fig: mpl.figure.Figure = plt.figure(dpi=dpi)
        self.ax: mpl.axes.Axes = self.fig.add_axes(
            [0, 0, 1, 1], projection=self.proj_map
        )

    def draw(
        self,
        extent: Optional[ExtentT] = None,
    ) -> RGBPlot:
        """Draw the plot."""
        self.ax.coastlines(color="white", resolution="50m", linewidth=0.75)
        self.ax.gridlines(color="white", linewidth=0.75, alpha=0.75)
        self.ax.imshow(self.arr, extent=self.data_extent, transform=self.proj_data)
        self.ax.set_extent(extent or self.data_extent, crs=self.proj_data)
        self.fig.canvas.draw()
        self.add_rgb_legend(n=2, labels_inside=True)
        return self

    def set_title(self, title: str, fontsize: Optional[float] = None) -> None:
        """Add main title to plot."""
        self.ax.set_title(title, fontsize=fontsize or self.font.large)

    def add_rgb_legend(
        self,
        n: int = 2,
        *,
        labels_inside: bool = True,
        gradients: bool = False,
    ) -> None:
        """Add RBG legend boxes next to the plot."""
        kwargs: dict[str, Any] = {
            "n": n,
            "rgb_order": "RGB",
            "rgb_labels": ("low", "mid", "high"),
            "gradients": gradients,
            "labels_inside": labels_inside,
        }
        if n == 3 and not labels_inside:
            kwargs["size"] = 0.25
            self._add_rgb_leg_tab(self, const_val=0.0, j=0, y0=0.125, **kwargs).draw()
            kwargs["label_x"] = False
            self._add_rgb_leg_tab(self, const_val=0.5, j=1, y0=0.425, **kwargs).draw()
            self._add_rgb_leg_tab(self, const_val=1.0, j=2, y0=0.725, **kwargs).draw()
        elif n == 2 and not labels_inside:
            kwargs["size"] = 0.35
            self._add_rgb_leg_tab(self, const_val=0.0, j=0, y0=0.120, **kwargs).draw()
            kwargs["label_x"] = False
            self._add_rgb_leg_tab(self, const_val=1.0, j=1, y0=0.570, **kwargs).draw()
        elif n == 2 and labels_inside:
            kwargs["size"] = 0.4
            self._add_rgb_leg_tab(self, const_val=0.0, j=0, y0=0.075, **kwargs).draw()
            self._add_rgb_leg_tab(self, const_val=1.0, j=1, y0=0.525, **kwargs).draw()
        else:
            raise NotImplementedError(f"n={n}; legend_labels_inside={labels_inside}")

    def _add_rgb_leg_tab(
        self,
        rgb_plot: RGBPlot,
        j: int,
        y0: float,
        size: float,
        **kwargs: Any,
    ) -> RGBLegendTable:
        """Add individual legend table to plot."""

        def get_legend_x0(ax: mpl.axes.Axes) -> float:
            """Derive x0 of legend table from map plot aspect ratio.

            Note that this formula has been derived from two data points (plots
            with different aspect ratios) and has not been tested thoroughly. It
            may have to be tweaked for plots with untested aspect ratios.

            The reason this formula is necessary in the first place is that it
            is not trivial to obtain the extent of cartopy map plots without
            fixed aspect ratio. For regular plots, this is easy with, e.g.,
            ``ax.get_position()``, but map plots are resized after the fact.

            """
            bbox = ax.get_window_extent()
            aspect: float = bbox.width / bbox.height
            x0 = (8.31 - aspect) * 0.142
            return x0

        tax = rgb_plot.fig.add_axes([0, j, 1, 1])
        x0 = get_legend_x0(rgb_plot.ax)
        tax.set_axes_locator(
            mplt.axes_grid1.inset_locator.InsetPosition(
                rgb_plot.ax, [x0, y0, size, size]
            )
        )
        return RGBLegendTable(tax, **kwargs)


class RGBLegendTable:
    """An RGB legend table to be added to an ``RGBPlot``."""

    def __init__(
        self,
        ax: mpl.axes.Axes,
        n: int,
        const_val: float,
        rgb_order: str = "RGB",
        *,
        gradients: bool = False,
        label_x: bool = True,
        labels_inside: bool = True,
        rgb_labels: tuple[str, str, str] = ("red", "green", "blue"),
        font: Optional[FontProperties] = None,
    ) -> None:
        """Create an instance of ``RGBLegendTable``.

        The x- and y-axis of the 2D box each corresponds to one RGB component,
        with the third component held constant.

        Args:
            ax: Axes of the table.

            n: Number of cols and rows.

            const_val: Constant value.

            rgb_order (optional): Combination of "R", "G" and "B"; the first/second
                second colors vary along the x/y axis of the table, while the third
                color has the constant value ``const_val``.

            gradients (optional): Show gradients between color cells instead of
                discrete cell boundaries.

            label_x (optional): Add x-axis label; has no effect if ``labels_inside``
                is true.

            labels_inside (optional): Show labels inside each cell instead of around
                the table.

            rgb_labels (optional): Labels corresponding to the three colors in order
                RGB; they are assigned in the order specified by ``rgb_order``.

            font (optional): Font properties.

        """
        if len(rgb_order) != 3 or set(rgb_order) != set("RGB"):
            raise ValueError(
                f"invalid order '{rgb_order}'; must be a combination of 'R', 'G', 'B'"
            )

        self.ax: mpl.axes.Axes = ax
        self.n: int = n
        self.const_val: float = const_val
        self.rgb_order: str = rgb_order
        self.gradients: bool = gradients
        self.label_x: bool = label_x
        self.labels_inside: bool = labels_inside
        self.font: FontProperties = not_none(font, FontProperties())

        self.nxy: int = 200 if self.gradients else self.n

        self.rgb_idcs: tuple[int, int, int] = (
            "RGB".index(self.rgb_order[0]),
            "RGB".index(self.rgb_order[1]),
            "RGB".index(self.rgb_order[2]),
        )
        self.rgb_labels: tuple[str, str, str] = (
            rgb_labels[self.rgb_idcs[0]],
            rgb_labels[self.rgb_idcs[1]],
            rgb_labels[self.rgb_idcs[2]],
        )

    def draw(self) -> RGBLegendTable:
        """Draw the color box."""
        # mypy 0.941 doesn't recognize type of np.array
        xlim: tuple[float, float] = (0.0 - 0.5, self.nxy - 0.5)
        ylim = xlim
        self.ax.set(xlim=xlim, ylim=ylim)
        self.ax.imshow(self._get_colors())
        line_color = "black"
        line_width = 0.5
        for spine in self.ax.spines.values():
            spine.set_color(line_color)
            spine.set_linewidth(line_width)
        if self.labels_inside:
            self._add_labels_inside()
        else:
            self._add_labels_outside()
        return self

    def _add_labels_outside(self) -> None:
        """Add labels outside the color boxes."""
        xlabel, ylabel, title = self.rgb_labels
        xyticks = np.arange(self.nxy)
        xyticklabels = list(map("{:0.1f}".format, xyticks / (self.nxy - 1)))
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")
        if self.label_x:
            self.ax.set_xticks(xyticks)
            self.ax.set_xticklabels(xyticklabels, fontsize=self.font.small)
        else:
            self.ax.set_xticks([])
        self.ax.set_yticks(xyticks)
        self.ax.set_yticklabels(xyticklabels, fontsize=self.font.small)
        self.ax.tick_params(axis="both", which="both", length=0)
        title += f" = {self.const_val:0.1f}"
        self.ax.text(
            s=title,
            x=-0.2,
            y=0.5,
            transform=self.ax.transAxes,
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=self.font.medium,
        )
        if self.label_x:
            self.ax.set_xlabel(xlabel, fontsize=self.font.medium)
        self.ax.set_ylabel(ylabel, fontsize=self.font.medium)

    def _add_labels_inside(self) -> None:
        """Add labels inside the color boxes."""
        xlabel, ylabel, title = self.rgb_labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        if self.n != 2:
            raise NotImplementedError(f"labels inside for n {self.n}, only for 2")
        colors_tab = self._get_colors_tab()
        # mypy/0.910 & numpy/1.21.4 error: "incompatible type; expected SupportsIndex"
        for j, i in np.ndindex(colors_tab.shape[:2]):
            r, g, b = colors_tab[j, i]
            color = bw_contrast(r, g, b)
            for k in [0, 1, 2]:
                if self.rgb_order.endswith("B"):
                    absent = [r, g, b][k] == 0
                else:
                    raise NotImplementedError(f"rgb_order '{self.rgb_order}'")
                label = ["low", "mid", "high"][k]
                label = f"{label}={0 if absent else 1}"
                f = self.nxy / self.n
                if not self.gradients:
                    # x = 0.5 * (f - 1) + f * i
                    # y = 0.5 * (f - 1) + f * (j + (k - 1) * 0.25)
                    # ha = "center"
                    x = 0.5 * (f - 1) + f * (i + (-1 if i == 0 else 1) * 0.35)
                    y = 0.5 * (f - 1) + f * (j + (k - 1) * 0.25)
                    ha = ["left", "right"][i]
                else:
                    x = 0.5 * (f - 1) + f * (i + (-1 if i == 0 else 1) * 0.4)
                    y = 0.5 * (f - 1) + f * (j + (k * 0.25 - (0.3 if j == 0 else 0.2)))
                    ha = ["left", "right"][i]
                self.ax.text(
                    x=x,
                    y=y,
                    s=label,
                    color=color,
                    ha=ha,
                    va="center",
                    fontsize=self.font.medium,
                    weight=None if absent else "bold",
                    alpha=0.66 if absent else 1.0,
                )

    def _get_colors(self) -> npt.NDArray[np.float_]:
        """Get colors."""
        if not self.gradients:
            return self._get_colors_tab()
        else:
            return self._get_colors_grad()

    def _get_colors_tab(self) -> npt.NDArray[np.float_]:
        """Get colors table."""
        colors_tab: npt.NDArray[np.float_]
        if self.n == 2:
            colors_tab = np.array(
                [
                    [(0.0, 0.0, self.const_val), (0.0, 1.0, self.const_val)],
                    [(1.0, 0.0, self.const_val), (1.0, 1.0, self.const_val)],
                ]
            )
        elif self.n == 3:
            colors_tab = np.array(
                [
                    [
                        (0.0, 0.0, self.const_val),
                        (0.0, 0.5, self.const_val),
                        (0.0, 1.0, self.const_val),
                    ],
                    [
                        (0.5, 0.0, self.const_val),
                        (0.5, 0.5, self.const_val),
                        (0.5, 1.0, self.const_val),
                    ],
                    [
                        (1.0, 0.0, self.const_val),
                        (1.0, 0.5, self.const_val),
                        (1.0, 1.0, self.const_val),
                    ],
                ]
            )
        else:
            raise ValueError(f"invalid n {self.n}; choices: 2, 3")
        # mypy 0.941 thinks return value has type Any (numpy 1.22.3)
        return colors_tab[:, :, (self.rgb_idcs[1], self.rgb_idcs[0], self.rgb_idcs[2])]

    def _get_colors_grad(self) -> npt.NDArray[np.float_]:
        """Get gradient colors."""
        if self.n != 2:
            raise NotImplementedError(f"gradients for n {self.n}")
        colors: npt.NDArray[np.float_] = np.empty([self.nxy, self.nxy, 3], np.float32)
        if self.rgb_order.endswith("B"):
            colors[:, :, 2] = self.const_val
            for i, j in np.ndindex(self.nxy, self.nxy):
                colors[i, j, 0] = logistic(i / (self.nxy - 1), 0.5, 20)
                colors[i, j, 1] = logistic(j / (self.nxy - 1), 0.5, 20)
        else:
            raise NotImplementedError(f"gradients for rgb_order '{self.rgb_order}'")
        return colors


class RGBDiffPlot:
    """Multipanel plot showing the differences between two RGB plots."""

    def __init__(
        self,
        arr1: npt.NDArray[np.float_],
        arr2: npt.NDArray[np.float_],
        *,
        data_extent: ExtentT,
        pollat: float,
        pollon: float,
        extent: Optional[ExtentT] = None,
        name1: str = "",
        name2: str = "",
        figsize: tuple[float, float] = (10, 6),
        font: Optional[FontProperties] = None,
    ) -> None:
        """Create an instance of ``RGBDiffPlot``."""
        self.arr1: npt.NDArray[np.float_] = arr1
        self.arr2: npt.NDArray[np.float_] = arr2
        self.data_extent: ExtentT = data_extent
        self.extent: ExtentT = not_none(extent, data_extent)
        self.name1: str = name1
        self.name2: str = name2
        self.font: FontProperties = not_none(font, FontProperties().bigger(2))

        self.proj_geo = ccrs.PlateCarree()
        self.proj_data = ccrs.RotatedPole(pole_latitude=pollat, pole_longitude=pollon)
        self.proj_map = self.proj_data

        self.fig: plt.figure.Figure = plt.figure(
            dpi=300,
            figsize=figsize,
            constrained_layout=True,
            facecolor="white",
        )
        self.map_axs: npt.NDArray[plt.axes.Axes] = np.full([2, 2], None, plt.Axes)
        self.leg_axs: npt.NDArray[plt.axes.Axes] = np.full([2], None, plt.Axes)
        self._add_axs()

    def draw(self) -> RGBDiffPlot:
        """Draw plot."""
        title = f"[a] {self.name1}\n[b] {self.name2}"
        self._add_fig_title(title)

        arrs = [
            self.arr1,
            self.arr2,
            np.where(self.arr1 > self.arr2, self.arr1 - self.arr2, 0.0),
            np.where(self.arr2 > self.arr1, self.arr2 - self.arr1, 0.0),
        ]
        titles = [
            "[a]",
            "[b]",
            "[b-a] removal",
            "[b-a] addition",
        ]
        arrs_iter = iter(arrs)
        titles_iter = iter(titles)
        for (i, j), ax in np.ndenumerate(self.map_axs):
            ax.set_extent(self.extent)
            ax.coastlines(color="white", resolution="50m", linewidth=0.75)
            ax.gridlines(color="white", linewidth=0.75, alpha=0.75)
            ax.imshow(
                next(arrs_iter),
                extent=self.data_extent,
                transform=self.proj_data,
            )
            self._add_ax_title(ax, next(titles_iter))

        self._add_legend()

        self.fig.canvas.draw()
        return self

    def plot(
        self,
        # mypy/0.910 w/ numpy/1.21.4 fails at nested ArrayLike with dtype
        # x: Union[float, npt.ArrayLike[np.floating]],
        # y: Union[float, npt.ArrayLike[np.floating]],
        x: Union[float, npt.ArrayLike],
        y: Union[float, npt.ArrayLike],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
    ]:
        """Call ``ax.plot`` on all four map axes."""
        ps: list[list[mpl.lines.Line2D]] = []
        for ax in self.map_axs.flatten():
            p = ax.plot(x, y, *args, **kwargs)
            ps.append(p)
        result = tuple(ps)
        assert len(result) == 4  # mypy
        result = cast(
            tuple[
                list[mpl.lines.Line2D],
                list[mpl.lines.Line2D],
                list[mpl.lines.Line2D],
                list[mpl.lines.Line2D],
            ],
            result,
        )  # mypy
        return result

    def plot_geo(
        self,
        # mypy/0.910 w/ numpy/1.21.4 fails at nested ArrayLike with dtype
        # lat: Union[float, npt.ArrayLike[np.floating]],
        # lon: Union[float, npt.ArrayLike[np.floating]],
        lat: Union[float, npt.ArrayLike],
        lon: Union[float, npt.ArrayLike],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
        list[mpl.lines.Line2D],
    ]:
        """Call ``ax.plot`` on all four map axes in geographical coordinates."""
        if "transform" in kwargs:
            raise TypeError("invalid keyword argument 'transform'")
        return self.plot(lon, lat, *args, transform=self.proj_geo, **kwargs)

    def _add_axs(self) -> None:
        """Add Axes to Figure with gridspec.

        Both ``self.fig`` and ``self.axs{...}`` must already be initialized.

        """
        heights = [2, 3, 3, 2]
        gs = self.fig.add_gridspec(4, 3, height_ratios=heights)

        # Add map axes
        axs = self.map_axs
        axs[0, 0] = self.fig.add_subplot(gs[0:2, 0], projection=self.proj_map)
        axs[0, 1] = self.fig.add_subplot(gs[0:2, 1], projection=self.proj_map)
        axs[1, 0] = self.fig.add_subplot(gs[2:4, 0], projection=self.proj_map)
        axs[1, 1] = self.fig.add_subplot(gs[2:4, 1], projection=self.proj_map)
        axs[0, 0].set_anchor("SE")
        axs[0, 1].set_anchor("S")
        axs[1, 0].set_anchor("NE")
        axs[1, 1].set_anchor("N")

        # Add legend axes
        axs = self.leg_axs
        axs[0] = self.fig.add_subplot(gs[1, 2])
        axs[1] = self.fig.add_subplot(gs[2, 2])
        axs[0].set_anchor("SW")
        axs[1].set_anchor("NW")

        # Add dummy axes above and below the legend axes
        for i in [0, 3]:
            dummy_ax = self.fig.add_subplot(gs[i])
            dummy_ax.set_visible(False)

    def _add_fig_title(self, s: str) -> None:
        """Add figure title."""
        self.fig.suptitle(
            s,
            fontsize=self.font.large,
            y=1.33,
            ha="left",
            x=0.03,
            transform=self.map_axs[0, 0].transAxes,
        )

    def _add_ax_title(self, ax: plt.axes.Axes, s: str) -> None:
        """Add title ``s`` to map axes ``(i, j)``."""
        ax.text(
            0.5,
            1.04,
            s,
            transform=ax.transAxes,
            ha="center",
            fontsize=self.font.medium,
        )

    def _add_legend(self) -> None:
        kwargs: dict[str, Any] = {
            "n": 2,
            "rgb_order": "RGB",
            "rgb_labels": ("low", "mid", "high"),
            "gradients": False,
            "labels_inside": True,
            "font": self.font.smaller(1),
        }
        RGBLegendTable(
            self.leg_axs[0],
            const_val=1.0,
            label_x=True,
            **kwargs,
        ).draw()
        RGBLegendTable(
            self.leg_axs[1],
            const_val=0.0,
            label_x=False,
            **kwargs,
        ).draw()
