"""Test string formatting of criteria."""
from __future__ import annotations

# Third-party
import pytest

# First-party
from atmcirclib.cosmo import COSMOGridDataset
from atmcirclib.traj.criteria import AllCriterion
from atmcirclib.traj.criteria import BoundaryZoneCriterion
from atmcirclib.traj.criteria import LeaveDomainCriterion
from atmcirclib.traj.criteria import VariableCriterion
from atmcirclib.traj.criteria import VariableCriterionFormatter

# pylint: disable=R0201  # no-self-use


class Test_CriterionFormat:
    """Format individual criterion instances."""

    # pylint: disable=W0212  # protected-access (Criterion._format_*)

    def test_all(self) -> None:
        """Format all-criterion."""
        crit = AllCriterion()
        assert crit._format_human() == "all"
        assert crit._format_file() == "all"

    def test_all_inv(self) -> None:
        """Format inverted all-criterion."""
        crit = AllCriterion().invert()
        assert crit._format_human() == "none"
        assert crit._format_file() == "none"

    def test_variable_closed(self) -> None:
        """Format variable criterion with closed range."""
        crit = VariableCriterion(
            variable="Z",
            time_idx=3,
            vmin=6000,
            vmax=9000,
        )
        assert crit._format_human() == "Z in 6000 to 9000"
        assert crit._format_file() == "z-6000-to-9000"

    def test_variable_closed_inv(self) -> None:
        """Format inverted variable criterion with closed range."""
        crit = VariableCriterion(
            variable="Z",
            time_idx=3,
            vmin=6000,
            vmax=9000,
        ).invert()
        assert crit._format_human() == "not Z in 6000 to 9000"
        assert crit._format_file() == "not-z-6000-to-9000"

    def test_variable_lower(self) -> None:
        """Format variable criterion with lower bound only."""
        crit = VariableCriterion(
            variable="UV",
            time_idx=3,
            vmin=30,
            vmax=None,
        )
        assert crit._format_human() == "UV >= 30"
        assert crit._format_file() == "uv-ge-30"

    def test_variable_upper(self) -> None:
        """Format variable criterion with upper bound only."""
        crit = VariableCriterion(
            variable="P",
            time_idx=3,
            vmin=None,
            vmax=1000,
        )
        assert crit._format_human() == "P <= 1000"
        assert crit._format_file() == "p-le-1000"

    def test_variable_none_fail(self) -> None:
        """Formatting variable criterion with no bounds raises an exception."""
        crit = VariableCriterion(
            variable="T",
            time_idx=3,
            vmin=None,
            vmax=None,
        )
        with pytest.raises(ValueError):
            crit._format_human()
        with pytest.raises(ValueError):
            crit._format_file()

    def test_leaving_domain(self) -> None:
        """Format leave-domain criterion."""
        crit = LeaveDomainCriterion()
        assert crit._format_human() == "leaving domain"
        assert crit._format_file() == "leaving-domain"

    def test_leaving_domain_inv(self) -> None:
        """Format inverted leave-domain criterion."""
        crit = LeaveDomainCriterion().invert()
        assert crit._format_human() == "never leaving domain"
        assert crit._format_file() == "never-leaving-domain"

    def test_boundary_zone(self) -> None:
        """Format boundary zone criterion."""
        grid: COSMOGridDataset = None  # type: ignore
        crit = BoundaryZoneCriterion(grid=grid, size_deg=1)
        assert crit._format_human() == "in boundary zone"
        assert crit._format_file() == "in-boundary-zone"

    def test_boundary_zone_inv(self) -> None:
        """Format inverted boundary zone criterion."""
        grid: COSMOGridDataset = None  # type: ignore
        crit = BoundaryZoneCriterion(grid=grid, size_deg=1).invert()
        assert crit._format_human() == "never in boundary zone"
        assert crit._format_file() == "never-in-boundary-zone"


class Test_VariableCriterionFormatter:
    """Use more info in variable criterion formatting with custom formatter."""

    def test_units(self) -> None:
        """Provide additional units."""
        crit = VariableCriterion(
            variable="Z",
            time_idx=3,
            vmin=6000,
            vmax=9000,
        )
        crit.formatter.units = "m"
        assert crit._format_human() == "Z in 6000 m to 9000 m"
        assert crit._format_file() == "z-6000m-to-9000m"

    def test_units_rel_time(self) -> None:
        """Provide additional units and relative time."""
        crit = VariableCriterion(
            variable="UV",
            time_idx=3,
            vmin=30,
            vmax=None,
        )
        crit.formatter = VariableCriterionFormatter(
            units="m/s",
            time=6,
            time_units="h",
            time_relative=True,
        )
        assert crit._format_human() == "UV @ +6 h >= 30 m/s"
        assert crit._format_file() == "uv@+6h-ge-30ms-1"

    def test_abs_time(self) -> None:
        """Format variable criterion with upper bound only."""
        crit = VariableCriterion(
            variable="P",
            time_idx=3,
            vmin=None,
            vmax=1000,
        )
        crit.formatter.time = 1
        crit.formatter.time_units = "h"
        crit.formatter.time_relative = False
        assert crit._format_human() == "P @ 1 h <= 1000"
        assert crit._format_file() == "p@1h-le-1000"
