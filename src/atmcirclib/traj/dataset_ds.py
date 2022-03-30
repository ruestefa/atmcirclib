"""Create and manipulate dataset corresponding to trajectories output files."""
from __future__ import annotations

# Standard library
from collections.abc import Mapping
from typing import Any
from typing import Optional
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


class TrajDatasetDsFactory:
    """Create a trajectories file dataset in COSMO output format."""

    # pylint: disable=R0902  # too-many-instance-attributes (>7)

    def __init__(
        self,
        *,
        dims: Optional[tuple[str, str]] = None,
        coords_d: Optional[
            Union[
                dict[str, list[float]],
                dict[str, npt.NDArray[np.generic]],
            ]
        ] = None,
        data_d: Optional[
            Union[
                dict[str, list[list[float]]],
                dict[str, npt.NDArray[np.generic]],
            ]
        ] = None,
        dtype_d: Optional[Mapping[str, npt.DTypeLike]] = None,
        scale_fact_d: Optional[Mapping[str, float]] = None,
        attrs_d: Optional[Mapping[str, Mapping[str, Any]]] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        vnan: Optional[float] = None,
    ) -> None:
        """Create a new instance."""
        self.dims: tuple[str, str] = dims or ("time", "id")
        self.coords_d: dict[str, npt.NDArray[np.generic]]
        self.data_d: dict[str, npt.NDArray[np.generic]]
        self.dtype_d: dict[str, npt.DTypeLike] = dict(dtype_d or {})
        self.scale_fact_d: dict[str, float] = dict(scale_fact_d or {})
        self.attrs_d: dict[str, dict[str, Any]] = {
            n: dict(v) for n, v in (attrs_d or {}).items()
        }
        self.attrs: dict[str, Any] = dict(attrs or {})
        self.vnan: float = vnan if vnan is not None else -999.0
        self.coords_d = self._prepare_arrs(coords_d) if coords_d else {}
        self.data_d = self._prepare_arrs(data_d) if data_d else {}

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

    def _prepare_arrs(
        self,
        raw_arrs: Union[
            Mapping[str, list[float]],
            Mapping[str, list[list[float]]],
            Mapping[str, npt.NDArray[np.generic]],
        ],
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Turn raw value lists into properly scaled data arrays."""
        arrs_by_name: dict[str, npt.NDArray[np.generic]] = {}
        arr: npt.NDArray[np.generic]
        for name, data in raw_arrs.items():
            if isinstance(data, np.ndarray):
                arr = data
            else:
                # pylint: disable=E1101  # no-member ('Field'.get)
                arr = np.array(data, self.dtype_d.get(name, np.float32))
            # pylint: disable=E1135  # unsupported-membership-test (scale_fact_d)
            if name in self.scale_fact_d:
                arr = np.where(
                    # pylint: disable=E1136  # unsubscriptable-object (scale_fact_d)
                    arr == self.vnan,
                    self.vnan,
                    arr * self.scale_fact_d[name],
                )
            arrs_by_name[name] = arr
        return arrs_by_name
