"""Procedures and data shared by ``test_traj_dataset`` modules.``."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Any
from typing import cast
from typing import Optional
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import xarray as xr

# Local
from ...test_cosmo.test_cosmo_grid_dataset import create_cosmo_grid_dataset_ds

__all__: list[str] = [
    "TrajsDatasetDsFactory",
    "create_cosmo_grid_dataset_ds",
]


@dc.dataclass()
class TrajsDatasetDsFactory:
    """Create trajs xr datasets for testing."""

    attrs: dict[str, Any]
    raw_coords_d: dict[str, list[float]]
    raw_data_d: dict[str, list[list[float]]]
    attrs_d: dict[str, dict[str, str]]
    dtype_d: dict[str, npt.DTypeLike]
    scale_fact_d: dict[str, float]
    vnan: float

    def __post_init__(self) -> None:
        """Finalize initialization."""
        self.ref_data_d: dict[str, npt.NDArray[np.generic]] = self._prepare_ref_arrs(
            self.raw_data_d
        )
        self.ref_coords_d: dict[str, npt.NDArray[np.generic]] = self._prepare_ref_arrs(
            self.raw_coords_d
        )

    def run(
        self,
        attrs: Optional[dict[str, Any]] = None,
        raw_coords_d: Optional[dict[str, npt.NDArray[np.generic]]] = None,
        raw_data_d: Optional[dict[str, npt.NDArray[np.generic]]] = None,
        attrs_d: Optional[dict[str, dict[str, str]]] = None,
    ) -> xr.Dataset:
        """Return a new dataset, optionally with changed arguments."""
        if attrs is None:
            attrs = self.attrs
        if raw_coords_d is None:
            coords_d = self.ref_coords_d
        else:
            coords_d = self._prepare_ref_arrs(raw_coords_d)
        if raw_data_d is None:
            data_d = self.ref_data_d
        else:
            data_d = self._prepare_ref_arrs(raw_data_d)
        if attrs_d is None:
            attrs_d = self.attrs_d
        return self._create_trajs_xr_dataset(
            attrs=attrs,
            coords_d=coords_d,
            data_d=data_d,
            attrs_d=attrs_d,
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
            arr = np.array(raw_data, self.dtype_d.get(name, np.float32))
            if name in self.scale_fact_d:
                arr = np.where(
                    arr == self.vnan, self.vnan, arr * self.scale_fact_d[name]
                )
            ref_arrs[name] = arr
        return ref_arrs

    @staticmethod
    def _create_trajs_xr_dataset(
        *,
        attrs: dict[str, Any],
        coords_d: dict[str, npt.NDArray[np.generic]],
        data_d: dict[str, npt.NDArray[np.generic]],
        attrs_d: dict[str, dict[str, str]],
    ) -> xr.Dataset:
        """Create a mock trajs xarray dataset as read from a NetCDF file."""

        if n := len(coords_d) != 1:
            raise NotImplementedError(
                f"{n} coords: " + ", ".join(map("'{}'".format, coords_d))
            )
        coord_name = next(iter(coords_d))
        dims = (coord_name, "id")

        def create_coord(name: str) -> xr.DataArray:
            """Create coordinate variable."""
            assert coords_d is not None  # mypy
            assert attrs_d is not None  # mypy
            return xr.DataArray(
                data=coords_d[name].copy(),
                coords={name: coords_d[name].copy()},
                dims=(name,),
                name=name,
                attrs=dict(attrs_d[name]),
            )

        def create_variable(name: str) -> xr.DataArray:
            """Create variable data array."""
            assert data_d is not None  # mypy
            assert attrs_d is not None  # mypy
            return xr.DataArray(
                data=data_d[name].copy(),
                dims=dims,
                name=name,
                attrs=dict(attrs_d[name]),
            )

        return xr.Dataset(
            coords={name: create_coord(name) for name in coords_d},
            data_vars={name: create_variable(name) for name in data_d},
            attrs=dict(attrs),
        )
