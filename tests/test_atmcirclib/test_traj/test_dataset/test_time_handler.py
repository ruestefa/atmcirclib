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
    # pylint: disable=R0913  # too-many-arguments (>5)
    # pylint: disable=W0622  # redefined-builtin (min)
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


class Test_GetTrajsStart:
    """Get trajectories start datetime."""

    def test_forward(self) -> None:
        """Forward trajs with unspecified model."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref, steps=np.arange(11), dtype="[m]", step_sec=60, dur_sec=600
        )
        ref_dt = dt.datetime(*ref)
        assert th.get_trajs_start() == ref_dt

    def test_forward_cosmo(self) -> None:
        """Forward trajs with COSMO, with zeroth-step-fix.

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
            steps=np.arange(11),
            dtype="[m]",
            step_sec=60,
            dur_sec=600,
            model="cosmo",
        )
        ref_dt = dt.datetime(*ref) + dt.timedelta(seconds=60)
        assert th.get_trajs_start() == ref_dt

    def test_forward_lagranto(self) -> None:
        """Forward trajs with LAGRANTO.

        The result is the same as w/o specifying a model.

        """
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(11),
            dtype="[m]",
            step_sec=60,
            dur_sec=600,
            model="lagranto",
        )
        ref_dt = dt.datetime(*ref)
        assert th.get_trajs_start() == ref_dt

    def test_backward(self) -> None:
        """Backward trajs with LAGRANTO."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=-np.arange(11),
            dtype="[m]",
            step_sec=-60,
            dur_sec=-600,
            model="lagranto",
        )
        ref_dt = dt.datetime(*ref)
        assert th.get_trajs_start() == ref_dt

    def test_backward_cosmo_fail(self) -> None:
        """Backward trajs with COSMO, which are impossible."""
        with pytest.raises(ValueError):
            create_time_handler(
                1990,
                2,
                steps=-np.arange(11),
                dtype="[m]",
                step_sec=-60,
                dur_sec=-600,
                model="cosmo",
            )


class Test_GetAbsSteps:
    """Get given steps as absolute datetimes."""

    def test_forward(self) -> None:
        """Forward trajs."""
        ref = (1990, 2, 24, 21)
        steps = np.arange(11)
        dtype = "timedelta64[h]"
        th = create_time_handler(
            *ref, steps=steps, dtype=dtype, step_sec=3600, dur_sec=36_000
        )
        ref_dt = dt.datetime(*ref)
        ref_steps = (ref_dt + steps.astype(dtype).astype(dt.timedelta)).tolist()
        assert ref_steps[0] == ref_dt
        assert th.get_abs_steps() == ref_steps
        assert th.get_abs_steps(slice(None, None, 2)) == ref_steps[slice(None, None, 2)]
        with pytest.raises(TypeError):
            th.get_abs_steps(1)  # type: ignore

    def test_forward_later(self) -> None:
        """Forward trajs, starting later than model."""
        ref = (1990, 2, 24, 21)
        steps = np.arange(100, 111)
        dtype = "timedelta64[h]"
        th = create_time_handler(
            *ref, steps=steps, dtype=dtype, step_sec=3600, dur_sec=36_000
        )
        ref_dt = dt.datetime(*ref)
        ref_steps = (ref_dt + steps.astype(dtype).astype(dt.timedelta)).tolist()
        assert ref_steps[0] != ref_dt
        assert th.get_abs_steps() == ref_steps
        assert th.get_abs_steps([7]) == ref_steps[7:8]

    def test_backward(self) -> None:
        """Backward trajs."""
        ref = (1990, 2, 24, 21)
        steps = -np.arange(50, 71, 2)
        dtype = "timedelta64[m]"
        th = create_time_handler(
            *ref, steps=steps, dtype=dtype, step_sec=-120, dur_sec=-1200
        )
        ref_dt = dt.datetime(*ref)
        ref_steps = (ref_dt + steps.astype(dtype).astype(dt.timedelta)).tolist()
        assert ref_steps[0] != ref_dt
        assert th.get_abs_steps() == ref_steps
        assert th.get_abs_steps([4, 7, 9]) == np.array(ref_steps)[[4, 7, 9]].tolist()


class Test_GetHoursSinceStart:
    """Get the time since start at a given step in hours."""

    def test_forward_full(self) -> None:
        """Forward trajs, targeting full hours only."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(60, 241, 20),
            dtype="[m]",
            step_sec=20 * 60,
            dur_sec=20 * 60 * 3,
        )
        assert th.get_hours_since_start(0) == 0
        assert th.get_hours_since_start(3) == 1
        assert th.get_hours_since_start(6) == 2
        assert th.get_hours_since_start(9) == 3
        assert th.get_hours_since_start(-1) == 3

    def test_forward_fraction(self) -> None:
        """Forward trajs, targeting any steps."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(60, 241, 20),
            dtype="[m]",
            step_sec=20 * 60,
            dur_sec=20 * 60 * 3,
        )
        assert th.get_hours_since_start(1) == 1 / 3
        assert th.get_hours_since_start(2) == 2 / 3
        assert th.get_hours_since_start(-2) == 2 + 2 / 3

    def test_backward(self) -> None:
        """Backward trajs."""
        ref = (1990, 2, 24, 21)
        th = create_time_handler(
            *ref,
            steps=np.arange(60, 241, 20)[::-1],
            dtype="[m]",
            step_sec=20 * 60,
            dur_sec=20 * 60 * 3,
        )
        assert th.get_hours_since_start(0) == -0
        assert th.get_hours_since_start(3) == -1
        assert th.get_hours_since_start(6) == -2
        assert th.get_hours_since_start(9) == -3
        assert th.get_hours_since_start(-1) == -3
