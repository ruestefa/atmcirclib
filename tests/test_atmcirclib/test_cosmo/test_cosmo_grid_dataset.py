"""Test class ``COSMOGridDataset``."""
from __future__ import annotations

# Standard library
from typing import cast

# Third-party
import cartopy.crs as ccrs
import numpy as np
import numpy.typing as npt
import xarray as xr

# First-party
from atmcirclib.cosmo import COSMOGridDataset
from atmcirclib.cosmo import create_cosmo_grid_dataset_ds
from atmcirclib.geo import unrotate_coords

__all__: list[str] = [
    "create_cosmo_grid_dataset_ds",
]

# pylint: disable=R0201  # no-self-use

# mypy 0.941 thinks arange returns array of type signed integer (numpy 1.33.2)
RLON: npt.NDArray[np.float_] = cast(npt.NDArray[np.float32], np.arange(-10.0, 5.1, 1.0))
RLAT: npt.NDArray[np.float_] = cast(npt.NDArray[np.float32], np.arange(-5.0, 4.1, 1.0))
POLE_RLON: float = 178.0
POLE_RLAT: float = 30.0
HEIGHT_TOA_M: float = 22_000.0


def create_test_ds() -> xr.Dataset:
    """Create a mock dataset representing a COSMO file with grids."""
    return create_cosmo_grid_dataset_ds(
        rlon=RLON,
        rlat=RLAT,
        pole_rlon=POLE_RLON,
        pole_rlat=POLE_RLAT,
        height_toa_m=HEIGHT_TOA_M,
    )


def test_grid_xr_dataset() -> None:
    """Test initialization of grid xarray test dataset."""
    create_test_ds()


class Test_GetBbox:
    """Test methods ``get_bbox_??`` that return the domain's bounding boxes."""

    def test_xy(self) -> None:
        """Get horizontal (x, y) bounding box."""
        grid = COSMOGridDataset(create_test_ds())
        bbox = grid.get_bbox_xy()
        ref = (RLON[0], RLON[-1], RLAT[0], RLAT[-1])
        assert bbox == ref

    def test_xz(self) -> None:
        """Get vertical (x, z) bounding box."""
        grid = COSMOGridDataset(create_test_ds())
        bbox = grid.get_bbox_xz()
        ref = (RLON[0], RLON[-1], 0.0, HEIGHT_TOA_M)
        assert bbox == ref

    def test_xz_km(self) -> None:
        """Get vertical (x, z) bounding box with z in km."""
        grid = COSMOGridDataset(create_test_ds(), z_unit="km")
        bbox = grid.get_bbox_xz()
        ref = (RLON[0], RLON[-1], 0.0, HEIGHT_TOA_M / 1000)
        assert bbox == ref

    def test_yz(self) -> None:
        """Get vertical (x, z) bounding box."""
        grid = COSMOGridDataset(create_test_ds())
        bbox = grid.get_bbox_yz()
        ref = (RLAT[0], RLAT[-1], 0.0, HEIGHT_TOA_M)
        assert bbox == ref

    def test_yz_km(self) -> None:
        """Get vertical (x, z) bounding box with z in km."""
        grid = COSMOGridDataset(create_test_ds(), z_unit="km")
        bbox = grid.get_bbox_yz()
        ref = (RLAT[0], RLAT[-1], 0.0, HEIGHT_TOA_M / 1000)
        assert bbox == ref


class Test_GetProj:
    """Test method ``get_proj`` that returns the domain's projection."""

    def test_type(self) -> None:
        """Check the type of the projection object."""
        grid = COSMOGridDataset(create_test_ds())
        proj = grid.get_proj()
        assert isinstance(proj, ccrs.RotatedPole)

    def test_transform(self) -> None:
        """Use the projection to unrotate the rlat/rlon grid arrays."""
        grid = COSMOGridDataset(create_test_ds())
        lon, lat = unrotate_coords(
            rlon=grid.ds.rlon.data,
            rlat=grid.ds.rlat.data,
            proj_rot=grid.get_proj(),
            transpose=True,
        )
        assert np.allclose(lon, grid.ds.lon.data)
        assert np.allclose(lat, grid.ds.lat.data)
