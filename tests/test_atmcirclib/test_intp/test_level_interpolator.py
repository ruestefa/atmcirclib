"""Test class ``atmcirclib.intp.LevelInterpolator``."""

# Standard library
from typing import cast

# Third-party
import numpy as np
import numpy.typing as npt
import pytest

# First-party
from atmcirclib.intp import LevelInterpolator


class Test_MonoGrid:
    """Test with simple idealized grid that monotonically increases in z."""

    nx: int = 7
    ny: int = 8
    nz: int = 9
    shape2d: tuple[int, int] = (nx, ny)
    shape3d: tuple[int, int, int] = (nx, ny, nz)

    def init_grid_fld(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Initialize grid and field arrays."""
        grid: npt.NDArray[np.float_] = np.zeros(self.shape3d, np.float32)
        fld: npt.NDArray[np.float_] = np.zeros(self.shape3d, np.float32)
        for k in range(self.nz):
            fld[:, :, k] = k
            for i in range(self.nx):
                grid[i, :, k] = i + k
        return (grid, fld)

    def init_ref(
        self, grid: npt.NDArray[np.float_], fld: npt.NDArray[np.float_], lvl: float
    ) -> npt.NDArray[np.float_]:
        """Create reference field by simple manual interpolation."""
        nx, ny, nz = fld.shape
        idcs_below_k = np.where(grid < lvl, 1, 0).sum(axis=2) - 1
        idcs_above_k = idcs_below_k + 1
        idcs_below = (*np.ogrid[:nx, :ny], idcs_below_k.clip(max=nz - 1))
        idcs_above = (*np.ogrid[:nx, :ny], idcs_above_k.clip(max=nz - 1))
        grid_below = grid[idcs_below]
        grid_above = grid[idcs_above]
        fld_below = fld[idcs_below]
        fld_above = fld[idcs_above]
        fld_below[idcs_below_k < 0] = np.nan
        fld_above[idcs_above_k < 0] = np.nan
        fld_below[idcs_below_k >= nz] = np.nan
        fld_above[idcs_above_k >= nz] = np.nan
        d_below = lvl - grid_below
        d_above = grid_above - lvl
        d_tot = d_below + d_above
        intp = d_above / d_tot * fld_below + d_below / d_tot * fld_above
        return cast(npt.NDArray[np.float_], intp)

    def test_intermed_level(self) -> None:
        """Interpolate to an intermediate level with no NaNs."""
        grid, fld = self.init_grid_fld()
        lvl = 6.6
        ref = self.init_ref(grid, fld, lvl)
        assert not np.isnan(ref).any()
        intp = LevelInterpolator(grid)
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)

    def test_low_level(self) -> None:
        """Interpolate to a low level with some below-ground NaNs.."""
        grid, fld = self.init_grid_fld()
        lvl = 3.2
        ref = self.init_ref(grid, fld, lvl)
        assert np.isnan(ref).any()
        intp = LevelInterpolator(grid)
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)

    def test_high_level(self) -> None:
        """Interpolate to a high level with some above-top NaNs."""
        grid, fld = self.init_grid_fld()
        lvl = 9.7
        ref = self.init_ref(grid, fld, lvl)
        assert np.isnan(ref).any()
        intp = LevelInterpolator(grid)
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)

    def test_decrease(self) -> None:
        """Invert the grid in z so it monotonically decreases."""
        grid, fld = self.init_grid_fld()
        grid = grid[:, :, ::-1]
        lvl = 3.8
        ref = self.init_ref(grid, fld, lvl)
        assert np.isnan(ref).any()
        intp = LevelInterpolator(grid)
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)


class Test_NonMonoGrid:
    """Test with idealized grid that doesn't monotonically in-/decrease in z."""

    nx: int = 4
    ny: int = 4
    nz: int = 6
    shape2d: tuple[int, int] = (nx, ny)
    shape3d: tuple[int, int, int] = (nx, ny, nz)

    def init_grid_fld(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Initialize grid and field arrays.

        The grid mostly increases monotonically in z, but some columns feature
        local stagnation or decreases.

        """
        grid: npt.NDArray[np.float_] = np.zeros(self.shape3d, np.float32)
        fld: npt.NDArray[np.float_] = np.zeros(self.shape3d, np.float32)
        grid[:, 0] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grid[:, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grid[0, :] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grid[3, :] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grid[1, 1] = [1.2, 2.8, 2.8, 3.6, 4.5, 5.5]
        grid[1, 2] = [1.5, 3.0, 2.8, 3.8, 3.0, 5.0]
        grid[2, 1] = [1.0, 2.5, 3.9, 3.0, 5.0, 6.0]
        grid[2, 2] = [2.0, 1.5, 3.0, 2.5, 4.6, 6.0]
        grid[2, 3] = [2.0, 1.5, 3.0, 2.5, 4.6, 6.0]
        fld[:, :] = np.arange(self.nz) + 1.0
        return (grid, fld)

    def test_no_direction_fails(self) -> None:
        """Initialization fails without a specified direction."""
        grid, _ = self.init_grid_fld()
        with pytest.raises(ValueError):
            LevelInterpolator(grid)

    def test_up(self) -> None:
        """Perform upward interpolation."""
        grid, fld = self.init_grid_fld()
        lvl = 3.2
        ref: npt.NDArray[np.float_] = np.zeros(self.shape2d, np.float32)
        ref[:, 0] = 3.2
        ref[:, 3] = 3.2
        ref[0, :] = 3.2
        ref[3, :] = 3.2
        ref[1, 1] = 3.5
        ref[1, 2] = 3.4
        ref[2, 1] = 2.5
        ref[2, 2] = 4.0 + 1 / 3
        ref[2, 3] = 4.0 + 1 / 3
        intp = LevelInterpolator(grid, direction="up")
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)

    def test_down(self) -> None:
        """Perform upward interpolation."""
        grid, fld = self.init_grid_fld()
        lvl = 3.2
        ref: npt.NDArray[np.float_] = np.zeros(self.shape2d, np.float32)
        ref[:, 0] = 3.2
        ref[:, 3] = 3.2
        ref[0, :] = 3.2
        ref[3, :] = 3.2
        ref[1, 1] = 3.5
        ref[1, 2] = 5.1
        ref[2, 1] = 4.1
        ref[2, 2] = 4.0 + 1 / 3
        ref[2, 3] = 4.0 + 1 / 3
        intp = LevelInterpolator(grid, direction="down")
        intp_fld = intp.to_level(fld, lvl)
        assert np.allclose(intp_fld, ref, equal_nan=True)
