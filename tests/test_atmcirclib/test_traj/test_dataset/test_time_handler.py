"""Test traj dataset time handler."""
from __future__ import annotations

# Standard library
import datetime as dt
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
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    min: int = 0,
    sec: int = 0,
    *,
    steps: npt.ArrayLike,
    dtype: npt.DTypeLike = "timedelta64[m]",
    step_sec: float,
    dur_sec: float,
    model: Optional[str] = None,
) -> TrajTimeHandler:
    """Create a traj time handler from given time steps and attributes."""
    if isinstance(dtype, str) and dtype.startswith("[") and dtype.endswith("]"):
        # Shorthand for timedelta: Unit only, e.g., '[h]', '[m]', '[s]'
        dtype = f"timedelta64{dtype}"
    ds = create_traj_dataset_ds(
        dims=("time", "id"),
        coords_d={"time": steps},
        dtype_d={"time": dtype},
        # data_d={"z": np.ones([len(steps), 10])},
        attrs={
            "ref_year": year,
            "ref_month": month,
            "ref_day": day,
            "ref_hour": hour,
            "ref_min": min,
            "ref_sec": sec,
            "duration_in_sec": dur_sec,
            "pollon": 178.0,
            "pollat": 30.0,
            "output_timestep_in_sec": step_sec,
        },
    )
    return TrajTimeHandler(TrajDataset(ds, model=model))


class Test_GetStart:
    """Get start datetime."""

    def test_forward(self) -> None:
        """Get start time of forward simulation."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref, steps=np.arange(10), dtype="[m]", step_sec=60, dur_sec=3600
        )
        ref_dt = dt.datetime(*ref)
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
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(10),
            dtype="[m]",
            step_sec=60,
            dur_sec=3600,
            model="cosmo",
        )
        ref_dt = dt.datetime(*ref) + dt.timedelta(seconds=60)
        assert th.get_start() == ref_dt

    def test_forward_lagranto(self) -> None:
        """Get start time of forward LAGRANTO simulation.

        The result is the same as w/o specifying a model.

        """
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(10),
            dtype="[m]",
            step_sec=60,
            dur_sec=3600,
            model="lagranto",
        )
        ref_dt = dt.datetime(*ref)
        assert th.get_start() == ref_dt

    def test_backward(self) -> None:
        """Get start time of backward simulation."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref, steps=-np.arange(10), dtype="[m]", step_sec=-60, dur_sec=-3600
        )
        ref_dt = dt.datetime(*ref)
        assert th.get_start() == ref_dt

    def test_backward_cosmo_fail(self) -> None:
        """Backward simulation with COSMO is not possible."""
        with pytest.raises(ValueError):
            create_time_handler(
                1990,
                2,
                steps=-np.arange(10),
                dtype="[m]",
                step_sec=-60,
                dur_sec=-3600,
                model="cosmo",
            )
