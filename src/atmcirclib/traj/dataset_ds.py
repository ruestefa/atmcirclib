"""Create and manipulate dataset corresponding to trajectories output files."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Any
from typing import cast
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr


def create_traj_dataset_ds(**kwargs: Any) -> xr.Dataset:
    """Create a trajectories file dataset in COSMO output format.

    See docstring of ``TrajDatasetDsFactory`` for possible arguments.

    """
    return TrajDatasetDsFactory(**kwargs).run()


@dc.dataclass()
class TrajDatasetDsFactory:
    """Create a trajectories file dataset in COSMO output format."""

    attrs: dict[str, Any] = dc.field(default_factory=dict)
    dims: tuple[str, str] = ("time", "id")
    raw_coords_d: dict[str, list[float]] = dc.field(default_factory=dict)
    raw_data_d: dict[str, list[list[float]]] = dc.field(default_factory=dict)
    attrs_d: dict[str, dict[str, str]] = dc.field(default_factory=dict)
    dtype_d: dict[str, npt.DTypeLike] = dc.field(default_factory=dict)
    scale_fact_d: dict[str, float] = dc.field(default_factory=dict)
    vnan: float = -999.0

    def __post_init__(self) -> None:
        """Finalize initialization."""
        self.data_d: dict[str, npt.NDArray[np.generic]] = self._prepare_ref_arrs(
            self.raw_data_d
        )
        self.coords_d: dict[str, npt.NDArray[np.generic]] = self._prepare_ref_arrs(
            self.raw_coords_d
        )

    def run(self) -> xr.Dataset:
        """Return a new dataset, optionally with changed arguments."""
        return xr.Dataset(
            coords={name: self._create_coord(name) for name in self.coords_d},
            data_vars={name: self._create_variable(name) for name in self.data_d},
            attrs=dict(self.attrs),
        )

    def _create_coord(self, name: str) -> xr.DataArray:
        """Create coordinate variable."""
        if name not in self.dims:
            dims_fmtd = ", ".join(map("'{}'".format, self.dims))
            raise ValueError(f"invalid coord name '{name}': not in dims ({dims_fmtd})")
        return xr.DataArray(
            data=self.coords_d[name].copy(),
            coords={name: self.coords_d[name].copy()},
            dims=(name,),
            name=name,
            # pylint: disable=E1101  # no-member ('Field'.get)
            attrs=dict(self.attrs_d.get(name, {})),
        )

    def _create_variable(self, name: str) -> xr.DataArray:
        """Create variable data array."""
        return xr.DataArray(
            data=self.data_d[name].copy(),
            dims=self.dims,
            name=name,
            # pylint: disable=E1101  # no-member ('Field'.get)
            attrs=dict(self.attrs_d.get(name, {})),
        )

    def _prepare_ref_arrs(
        self,
        raw_arrs: Union[
            dict[str, list[float]],
            dict[str, list[list[float]]],
            dict[str, npt.NDArray[np.generic]],
        ],
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Turn raw value lists into properly scaled data arrays."""
        if not raw_arrs or isinstance(next(iter(raw_arrs.values())), np.ndarray):
            assert all(isinstance(v, np.ndarray) for v in raw_arrs.values())
            return {
                name: cast(npt.NDArray[np.generic], arr.copy())
                for name, arr in raw_arrs.items()
            }
        ref_arrs: dict[str, npt.NDArray[np.generic]] = {}
        arr: npt.NDArray[np.generic]
        for name, raw_data in raw_arrs.items():
            # pylint: disable=E1101  # no-member ('Field'.get)
            arr = np.array(raw_data, self.dtype_d.get(name, np.float32))
            # pylint: disable=E1135  # unsupported-membership-test (scale_fact_d)
            if name in self.scale_fact_d:
                arr = np.where(
                    # pylint: disable=E1136  # unsubscriptable-object (scale_fact_d)
                    arr == self.vnan,
                    self.vnan,
                    arr * self.scale_fact_d[name],
                )
            ref_arrs[name] = arr
        return ref_arrs
