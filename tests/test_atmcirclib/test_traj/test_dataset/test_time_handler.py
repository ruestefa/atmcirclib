"""Test traj dataset time handler."""
from __future__ import annotations

# Standard library
import datetime as dt
from typing import Any
from typing import Optional

# Third-party
import numpy as np
import numpy.typing as npt
import pytest

# First-party
from atmcirclib.traj.dataset import TrajDataset
from atmcirclib.traj.dataset import TrajTimeHandler

# Local
from .shared import create_traj_dataset_ds

# pylint: disable=R0201  # no-self-use


def create_time_handler(
    steps: npt.ArrayLike,
    dtype: npt.DTypeLike = "timedelta64[m]",
    *,
    step_sec: float,
    dur_sec: float,
    model: Optional[str] = None,
    **attrs: Any,
) -> TrajTimeHandler:
    """Create a traj time handler from given time steps and attributes."""
    if isinstance(dtype, str) and dtype.startswith("[") and dtype.endswith("]"):
        # Shorthand for timedelta: Unit only, e.g., '[h]', '[m]', '[s]'
        dtype = f"timedelta64{dtype}"
    default_attrs = {
        "ref_year": 2016,
        "ref_month": 9,
        "ref_day": 20,
        "ref_hour": 0,
        "ref_min": 0,
        "ref_sec": 0,
        "duration_in_sec": dur_sec,
        "pollon": 178.0,
        "pollat": 30.0,
        "output_timestep_in_sec": step_sec,
    }
    ds = create_traj_dataset_ds(
        dims=("time", "id"),
        coords_d={"time": steps},
        dtype_d={"time": dtype},
        # data_d={"z": np.ones([len(steps), 10])},
        attrs={**default_attrs, **attrs},
    )
    return TrajTimeHandler(TrajDataset(ds, model=model))


class Test_GetStart:
    """Get start datetime."""

    def test_forward(self) -> None:
        """Get start time of forward simulation."""
        ref_kw: dict[str, Any] = {
            "ref_year": 1990,
            "ref_month": 2,
            "ref_day": 24,
            "ref_hour": 21,
        }
        th = create_time_handler(
            np.arange(10), "[m]", step_sec=60, dur_sec=3600, **ref_kw
        )
        ref_dt = dt.datetime(*ref_kw.values())
        assert th.get_start() == ref_dt

    def test_forward_cosmo(self) -> None:
        """Get start time of forward COSMO simulation with start time fix.

        The fix constitutes skipping the zeroth time step, which has been
        manually added in order to add the start positions to the output file.

        (By default, the online traj module started writing only after the first
        advection step, so the original start positions were missing in the
        output. Currently (2022-03), while the start positions have been added,
        this is not cleanly implemented, e.g., the respective time step is one
        model time step prior to the start time, rather than one trajectory
        output time step, which differ for ``ninc_out_traj > 0``.)

        """
        ref_kw: dict[str, Any] = {
            "ref_year": 1990,
            "ref_month": 2,
            "ref_day": 24,
            "ref_hour": 21,
        }
        th = create_time_handler(
            np.arange(10), "[m]", step_sec=60, dur_sec=3600, model="cosmo", **ref_kw
        )
        ref_dt = dt.datetime(*ref_kw.values()) + dt.timedelta(seconds=60)
        assert th.get_start() == ref_dt

    def test_forward_lagranto(self) -> None:
        """Get start time of forward LAGRANTO simulation.

        The result is the same as w/o specifying a model.

        """
        ref_kw: dict[str, Any] = {
            "ref_year": 1990,
            "ref_month": 2,
            "ref_day": 24,
            "ref_hour": 21,
        }
        th = create_time_handler(
            np.arange(10), "[m]", step_sec=60, dur_sec=3600, model="lagranto", **ref_kw
        )
        ref_dt = dt.datetime(*ref_kw.values())
        assert th.get_start() == ref_dt

    def test_backward(self) -> None:
        """Get start time of backward simulation."""
        ref_kw: dict[str, Any] = {
            "ref_year": 1990,
            "ref_month": 2,
            "ref_day": 24,
            "ref_hour": 21,
        }
        th = create_time_handler(
            -np.arange(10), "[m]", step_sec=-60, dur_sec=-3600, **ref_kw
        )
        ref_dt = dt.datetime(*ref_kw.values())
        assert th.get_start() == ref_dt

    def test_backward_cosmo_fail(self) -> None:
        """Backward simulation with COSMO is not possible."""
        with pytest.raises(ValueError):
            create_time_handler(
                -np.arange(10), "[m]", step_sec=-60, dur_sec=-3600, model="cosmo"
            )
