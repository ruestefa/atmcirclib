"""Test some GT4Py stencils of increasing complexity."""
# Standard library
from typing import Any

# Third-party
import gt4py.gtscript as gts
import gt4py.storage as gt_store
import numpy as np
import numpy.typing as npt
from gt4py.gtscript import BACKWARD
from gt4py.gtscript import computation
from gt4py.gtscript import FORWARD
from gt4py.gtscript import interval
from gt4py.gtscript import isnan
from gt4py.gtscript import PARALLEL
from gt4py.gtscript import sqrt

BACKEND = "gtc:numpy"
DTYPE = np.float32


def test_copy() -> None:
    """Copy a field."""

    @gts.stencil(backend=BACKEND)
    def copy(
        in_fld: gts.Field[DTYPE],
        out_fld: gts.Field[DTYPE],
    ) -> None:
        """Copy the input field to the output field."""
        with computation(PARALLEL), interval(...):
            out_fld[...] = in_fld[0, 0, 0]

    # Define fields
    (nx, ny, nz) = shape3d = (4, 5, 6)
    in_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    for i in range(nx):
        for k in range(nz):
            in_data[i, :, k] = i + k
    out_data: npt.NDArray[np.float_] = np.zeros(in_data.shape, DTYPE)

    # Define stores
    in_store = gt_store.from_array(
        in_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )
    out_store = gt_store.from_array(
        out_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )

    # Make sure the fields differ
    assert not np.equal(in_store, out_store).all()

    # Run the test: Copy one field to the other
    copy(in_store, out_store)

    # Now the fields are equal
    assert np.equal(in_store, out_store).all()


def test_vertical_shift_periodic() -> None:
    """Shift a field upward twice, whereby the vertical axis acts as periodic."""

    @gts.stencil(backend=BACKEND)
    def shift_up(
        in_fld: gts.Field[gts.IJK, DTYPE],
        wk2d_fld: gts.Field[gts.IJ, DTYPE],
        out_fld: gts.Field[gts.IJK, DTYPE],
    ) -> None:
        """Shift field upward by one, moving the abovemost layer to the bottom."""
        with computation(FORWARD), interval(-1, None):
            wk2d_fld[...] = in_fld[0, 0, 0]
        with computation(FORWARD), interval(0, 1):
            out_fld[...] = wk2d_fld[0, 0]
        with computation(FORWARD), interval(1, None):
            out_fld[...] = in_fld[0, 0, -1]

    # Define fields
    (nx, ny, nz) = shape3d = *shape2d, _ = (4, 5, 6)
    in_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    for i in range(nx):
        for k in range(nz):
            in_data[i, :, k] = i + k
    out_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)

    # Define stores
    in_store = gt_store.from_array(
        in_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )
    wk2d_store = gt_store.empty(
        shape=shape2d, mask="IJ", backend=BACKEND, default_origin=(0, 0), dtype=DTYPE
    )
    out_store = gt_store.from_array(
        out_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )

    # Create test reference and make sure the output array doesn't validate yet
    ref: npt.NDArray[np.float_] = np.dstack([in_data[:, :, -2:], in_data[:, :, :-2]])
    assert not np.equal(in_store, ref).all()

    # Run test: Shift field upward twice
    shift_up(in_store, wk2d_store, out_store)
    shift_up(out_store, wk2d_store, in_store)

    # Now the output field validates
    assert np.equal(in_store, ref).all()


def test_vert_intp_nearest_neighbor() -> None:
    """Nearest-neighbor interpolation of a 3D field to a 2D surface."""

    @gts.stencil(backend=BACKEND)
    def interpolate(
        fld: gts.Field[gts.IJK, DTYPE],
        grid: gts.Field[gts.IJK, DTYPE],
        val: float,
        intp: gts.Field[gts.IJ, DTYPE],
        vnan: float,
    ) -> None:
        """Interpolate ``fld`` to ``val``-surface of ``grid``."""
        with computation(FORWARD), interval(-1, None):
            intp[...] = vnan
        with computation(FORWARD), interval(...):
            if grid == val:
                intp[...] = fld[0, 0, 0]

    # Define fields
    (nx, ny, nz) = shape3d = *shape2d, _ = (4, 5, 6)
    grid_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    in_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    for k in range(nz):
        in_data[:, :, k] = k
        for i in range(nx):
            grid_data[i, :, k] = i + k
    val = 6.0

    # Define stores
    in_store = gt_store.from_array(
        in_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )
    grid_store = gt_store.from_array(
        grid_data, backend=BACKEND, default_origin=(0, 0, 0), dtype=DTYPE
    )
    intp_store = gt_store.empty(
        shape=shape2d, mask="IJ", backend=BACKEND, default_origin=(0, 0), dtype=DTYPE
    )

    # Create test reference and ensure the output array doesn't validate yet
    idcs_k = np.where(grid_data < val, 1, 0).sum(axis=2)
    idcs = (*np.ogrid[:nx, :ny], idcs_k.clip(max=nz - 1))
    ref = in_data[idcs]
    ref[idcs_k >= nz] = np.nan
    assert not np.equal(intp_store, ref).all()

    # Run test: Interpolate field to surface
    interpolate(in_store, grid_store, val, intp_store, np.nan)

    # Now the output array validates
    assert np.allclose(intp_store.data, ref, equal_nan=True)


def test_vert_intp_linear() -> None:
    """Linear interpolation of a 3D field to a 2D surface."""

    @gts.stencil(backend=BACKEND)
    def interpolate(
        fld: gts.Field[gts.IJK, DTYPE],
        grid: gts.Field[gts.IJK, DTYPE],
        val: float,
        intp: gts.Field[gts.IJ, DTYPE],
        wk_below: gts.Field[gts.IJ, DTYPE],
        wk_above: gts.Field[gts.IJ, DTYPE],
        d_below: gts.Field[gts.IJ, DTYPE],
        d_above: gts.Field[gts.IJ, DTYPE],
        vnan: float,
    ) -> None:
        """Interpolate ``fld`` to ``val``-surface of ``grid``."""
        with computation(FORWARD), interval(...):
            wk_below[...] = vnan
            d_below[...] = 0.0
        with computation(FORWARD), interval(...):
            if grid < val:
                wk_below[...] = fld[0, 0, 0]
                d_below[...] = val - grid[0, 0, 0]
        with computation(BACKWARD), interval(...):
            wk_above[...] = vnan
            d_above[...] = 0.0
        with computation(BACKWARD), interval(...):
            if grid > val:
                wk_above[...] = fld[0, 0, 0]
                d_above[...] = grid[0, 0, 0] - val
        with computation(FORWARD), interval(...):
            if (
                isnan(vnan) and (isnan(wk_below[0, 0]) == vnan or isnan(wk_above[0, 0]))
            ) or (wk_below[0, 0] == vnan or wk_above[0, 0] == vnan):
                intp[...] = vnan
            else:
                intp[...] = (
                    d_above[0, 0] / (d_below[0, 0] + d_above[0, 0]) * wk_below[0, 0]
                    + d_below[0, 0] / (d_below[0, 0] + d_above[0, 0]) * wk_above[0, 0]
                )

    # Define fields
    (nx, ny, nz) = shape3d = *shape2d, _ = (4, 5, 6)
    grid_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    fld_data: npt.NDArray[np.float_] = np.zeros(shape3d, DTYPE)
    for k in range(nz):
        fld_data[:, :, k] = k
        for i in range(nx):
            grid_data[i, :, k] = i + k
    val = 5.2

    # Define stores
    kwargs3d: dict[str, Any] = {
        "default_origin": (0, 0, 0),
        "dtype": DTYPE,
        "backend": BACKEND,
    }
    kwargs2d: dict[str, Any] = {**kwargs3d, "default_origin": (0, 0), "mask": "IJ"}
    in_store = gt_store.from_array(fld_data, **kwargs3d)
    grid_store = gt_store.from_array(grid_data, **kwargs3d)
    intp_store = gt_store.empty(shape=shape2d, **kwargs2d)
    wk1_store = gt_store.empty(shape=shape2d, **kwargs2d)
    wk2_store = gt_store.empty(shape=shape2d, **kwargs2d)
    wk3_store = gt_store.empty(shape=shape2d, **kwargs2d)
    wk4_store = gt_store.empty(shape=shape2d, **kwargs2d)

    # Create test reference and ensure the output array doesn't validate yet
    idcs_below_k = np.where(grid_data < val, 1, 0).sum(axis=2) - 1
    idcs_above_k = idcs_below_k + 1
    idcs_below = (*np.ogrid[:nx, :ny], idcs_below_k.clip(max=nz - 1))
    idcs_above = (*np.ogrid[:nx, :ny], idcs_above_k.clip(max=nz - 1))
    grid_below = grid_data[idcs_below]
    grid_above = grid_data[idcs_above]
    ref_below = fld_data[idcs_below]
    ref_above = fld_data[idcs_above]
    ref_below[idcs_below_k < 0] = np.nan
    ref_above[idcs_above_k < 0] = np.nan
    ref_below[idcs_below_k >= nz] = np.nan
    ref_above[idcs_above_k >= nz] = np.nan
    d_below = val - grid_below
    d_above = grid_above - val
    d_tot = d_below + d_above
    ref = d_above / d_tot * ref_below + d_below / d_tot * ref_above
    assert not np.equal(intp_store, ref).all()

    # Run test: Interpolate field to surface
    interpolate(
        in_store,
        grid_store,
        val,
        intp_store,
        wk1_store,
        wk2_store,
        wk3_store,
        wk4_store,
        np.nan,
    )

    # Now the output array validates
    assert np.allclose(intp_store.data, ref, equal_nan=True)


def test_gradient() -> None:  # noqa: C901  # too complex
    """Compute the horizontal gradient of a 3D field."""

    def arr_store(
        arr: npt.NDArray[np.generic],
        origin: tuple[int, int, int] = (0, 0, 0),
        *,
        backend: str = BACKEND,
        dtype: npt.DTypeLike = DTYPE,
    ) -> gt_store.Storage:
        """Create a storage object from an array."""
        return gt_store.from_array(
            arr, default_origin=origin, dtype=dtype, backend=backend
        )

    def empty_store(
        shape: tuple[int, ...],
        origin: tuple[int, int, int] = (0, 0, 0),
        *,
        backend: str = BACKEND,
        dtype: npt.DTypeLike = DTYPE,
    ) -> gt_store.Storage:
        """Create a storage object from an array."""
        return gt_store.empty(
            shape=shape, default_origin=origin, dtype=dtype, backend=backend
        )

    def gradient(fld: npt.NDArray[np.float_], dxy: float) -> npt.NDArray[np.float_]:
        """Compute the horizontal gradient of a 3D field."""

        def gradient_x(
            fld: npt.NDArray[np.float_], dx: float
        ) -> npt.NDArray[np.float_]:
            """Compute the gradient in x-direction."""

            @gts.stencil(backend=BACKEND)
            def grad_x_fw(
                fld: gts.Field[gts.IJK, DTYPE],
                dx: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute forward horizontal gradient in x-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[1, 0, 0] - fld[0, 0, 0]) / dx

            @gts.stencil(backend=BACKEND)
            def grad_x_bw(
                fld: gts.Field[gts.IJK, DTYPE],
                dx: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute backward horizontal gradient in x-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[0, 0, 0] - fld[-1, 0, 0]) / dx

            @gts.stencil(backend=BACKEND)
            def grad_x_c(
                fld: gts.Field[gts.IJK, DTYPE],
                dx: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute central horizontal gradient in x-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[1, 0, 0] - fld[-1, 0, 0]) / (2 * dx)

            fld_x_fw_s = arr_store(fld)
            grad_x_fw_s = empty_store(shape3d)
            grad_x_fw(fld_x_fw_s, dxy, grad_x_fw_s)

            fld_x_bw_s = arr_store(fld, (1, 0, 0))
            grad_x_bw_s = empty_store(shape3d, (1, 0, 0))
            grad_x_bw(fld_x_bw_s, dxy, grad_x_bw_s)

            fld_x_c_s = arr_store(fld, (1, 0, 0))
            grad_x_c_s = empty_store(shape3d, (1, 0, 0))
            grad_x_c(fld_x_c_s, dxy, grad_x_c_s)

            grad_fld_x: npt.NDArray[np.float_] = np.empty(shape3d, DTYPE)
            grad_fld_x[:2, :, :] = grad_x_fw_s.data[:2, :, :]
            grad_fld_x[-2:, :, :] = grad_x_bw_s.data[-2:, :, :]
            grad_fld_x[1:-1, :, :] = grad_x_c_s.data[1:-1, :, :]

            return grad_fld_x

        def gradient_y(
            fld: npt.NDArray[np.float_], dy: float
        ) -> npt.NDArray[np.float_]:
            """Compute the gradient in y-direction."""

            @gts.stencil(backend=BACKEND)
            def grad_y_fw(
                fld: gts.Field[gts.IJK, DTYPE],
                dy: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute forward horizontal gradient in y-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[0, 1, 0] - fld[0, 0, 0]) / dy

            @gts.stencil(backend=BACKEND)
            def grad_y_bw(
                fld: gts.Field[gts.IJK, DTYPE],
                dy: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute backward horizontal gradient in y-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[0, 0, 0] - fld[0, -1, 0]) / dy

            @gts.stencil(backend=BACKEND)
            def grad_y_c(
                fld: gts.Field[gts.IJK, DTYPE],
                dy: float,
                grad: gts.Field[gts.IJK, DTYPE],
            ) -> None:
                """Compute central horizontal gradient in y-direction."""
                with computation(PARALLEL), interval(...):
                    grad[...] = (fld[0, 1, 0] - fld[0, -1, 0]) / (2 * dy)

            fld_y_fw_s = arr_store(fld)
            grad_y_fw_s = empty_store(shape3d, (0, 0, 0))
            grad_y_fw(fld_y_fw_s, dxy, grad_y_fw_s)

            fld_y_bw_s = arr_store(fld, (0, 1, 0))
            grad_y_bw_s = empty_store(shape3d, (0, 1, 0))
            grad_y_bw(fld_y_bw_s, dxy, grad_y_bw_s)

            fld_y_c_s = arr_store(fld, (0, 1, 0))
            grad_y_c_s = empty_store(shape3d, (0, 1, 0))
            grad_y_c(fld_y_c_s, dxy, grad_y_c_s)

            grad_fld_y: npt.NDArray[np.float_] = np.empty(shape3d, DTYPE)
            grad_fld_y[:, :2, :] = grad_y_fw_s.data[:, :2, :]
            grad_fld_y[:, -2:, :] = grad_y_bw_s.data[:, -2:, :]
            grad_fld_y[:, 1:-1, :] = grad_y_c_s.data[:, 1:-1, :]

            return grad_fld_y

        @gts.stencil(backend=BACKEND)
        def abs_grad(
            grad_x: gts.Field[gts.IJK, DTYPE],
            grad_y: gts.Field[gts.IJK, DTYPE],
            grad: gts.Field[gts.IJK, DTYPE],
        ) -> None:
            """Compute absolute horizontal gradient from its components."""
            with computation(PARALLEL), interval(...):
                grad[...] = sqrt(grad_x[0, 0, 0] ** 2 + grad_y[0, 0, 0] ** 2)

        grad_fld_x = gradient_x(fld, dxy)
        grad_fld_y = gradient_y(fld, dxy)

        grad_x_s = arr_store(grad_fld_x)
        grad_y_s = arr_store(grad_fld_y)
        grad_s = empty_store(shape3d)
        abs_grad(grad_x_s, grad_y_s, grad_s)

        return np.asarray(grad_s.data)

    # Define input field
    (nx, ny, nz) = shape3d = (4, 5, 2)
    fld: npt.NDArray[np.float_] = np.empty(shape3d, DTYPE)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fld[i, j, k] = i + j**2 + k**3
    dxy = 1.0

    # Compute reference gradient
    grad_x, grad_y, _ = np.gradient(fld)
    ref = np.sqrt(grad_x**2 + grad_y**2)

    # Compute and validate gradient
    grad_fld = gradient(fld, dxy)
    assert np.allclose(grad_fld, ref)
