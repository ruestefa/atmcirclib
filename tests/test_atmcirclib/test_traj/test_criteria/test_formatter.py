"""Test string formatting of criteria."""
from __future__ import annotations

# Third-party
import pytest

# First-party
from atmcirclib.cosmo import COSMOGridDataset
from atmcirclib.traj.criteria import AllCriterion
from atmcirclib.traj.criteria import BoundaryZoneCriterion
from atmcirclib.traj.criteria import Criteria
from atmcirclib.traj.criteria import CriteriaFormatter
from atmcirclib.traj.criteria import LeaveDomainCriterion
from atmcirclib.traj.criteria import VariableCriterion
from atmcirclib.traj.criteria import VariableCriterionFormatter

# pylint: disable=R0201  # no-self-use


class Test_CriterionFormat:
    """Format individual criterion instances."""

    def test_all(self) -> None:
        """Format all-criterion."""
        crit = AllCriterion()
        assert crit.format("human") == "all"
        assert crit.format("file") == "all"

    def test_all_inv(self) -> None:
        """Format inverted all-criterion."""
        crit = AllCriterion().invert()
        assert crit.format("human") == "none"
        assert crit.format("file") == "none"

    def test_variable_closed(self) -> None:
        """Format variable criterion with closed range."""
        crit = VariableCriterion(
            variable="Z",
            time_idx=3,
            vmin=6000,
            vmax=9000,
        )
        assert crit.format("human") == "Z in 6000 to 9000"
        assert crit.format("file") == "z-6000-to-9000"

    def test_variable_closed_inv(self) -> None:
        """Format inverted variable criterion with closed range."""
        crit = VariableCriterion(
            variable="Z",
            time_idx=3,
            vmin=6000,
            vmax=9000,
        ).invert()
        assert crit.format("human") == "not Z in 6000 to 9000"
        assert crit.format("file") == "not-z-6000-to-9000"

    def test_variable_lower(self) -> None:
        """Format variable criterion with lower bound only."""
        crit = VariableCriterion(
            variable="UV",
            time_idx=3,
            vmin=30,
            vmax=None,
        )
        assert crit.format("human") == "UV >= 30"
        assert crit.format("file") == "uv-ge-30"

    def test_variable_upper(self) -> None:
        """Format variable criterion with upper bound only."""
        crit = VariableCriterion(
            variable="P",
            time_idx=3,
            vmin=None,
            vmax=1000,
        )
        assert crit.format("human") == "P <= 1000"
        assert crit.format("file") == "p-le-1000"

    def test_variable_none_fail(self) -> None:
        """Formatting variable criterion with no bounds raises an exception."""
        crit = VariableCriterion(
            variable="T",
            time_idx=3,
            vmin=None,
            vmax=None,
        )
        with pytest.raises(ValueError):
            crit.format("human")
        with pytest.raises(ValueError):
            crit.format("file")

    def test_leaving_domain(self) -> None:
        """Format leave-domain criterion."""
        crit = LeaveDomainCriterion()
        assert crit.format("human") == "leaving domain"
        assert crit.format("file") == "leaving-domain"

    def test_leaving_domain_inv(self) -> None:
        """Format inverted leave-domain criterion."""
        crit = LeaveDomainCriterion().invert()
        assert crit.format("human") == "never leaving domain"
        assert crit.format("file") == "never-leaving-domain"

    def test_boundary_zone(self) -> None:
        """Format boundary zone criterion."""
        grid: COSMOGridDataset = None  # type: ignore
        crit = BoundaryZoneCriterion(grid=grid, size_deg=1)
        assert crit.format("human") == "in boundary zone"
        assert crit.format("file") == "in-boundary-zone"

    def test_boundary_zone_inv(self) -> None:
        """Format inverted boundary zone criterion."""
        grid: COSMOGridDataset = None  # type: ignore
        crit = BoundaryZoneCriterion(grid=grid, size_deg=1).invert()
        assert crit.format("human") == "never in boundary zone"
        assert crit.format("file") == "never-in-boundary-zone"


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
        assert crit.format("human") == "Z in 6000 m to 9000 m"
        assert crit.format("file") == "z-6000m-to-9000m"

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
        assert crit.format("human") == "UV @ +6 h >= 30 m/s"
        assert crit.format("file") == "uv@+6h-ge-30ms-1"

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
        assert crit.format("human") == "P @ 1 h <= 1000"
        assert crit.format("file") == "p@1h-le-1000"


class Test_CriteriaFormatter:
    """Format multiple criteria at once."""

    def test_base(self) -> None:
        """Use basic formatter for all criterion types."""
        grid: COSMOGridDataset = None  # type: ignore
        crits = Criteria(
            [
                LeaveDomainCriterion().invert(),
                BoundaryZoneCriterion(grid=grid, size_deg=1).invert(),
                VariableCriterion(
                    variable="Z",
                    time_idx=3,
                    vmin=6000,
                    vmax=9000,
                ),
                VariableCriterion(
                    variable="UV",
                    time_idx=3,
                    vmin=30,
                    vmax=None,
                ),
            ],
            require_all=True,
        )
        formatter = CriteriaFormatter()
        human_parts = [
            "never leaving domain",
            "never in boundary zone",
            "Z in 6000 to 9000",
            "UV >= 30",
        ]
        human = " and ".join(human_parts)
        assert formatter.format_human(crits) == human
        file_parts = [
            "never-leaving-domain",
            "never-in-boundary-zone",
            "z-6000-to-9000",
            "uv-ge-30",
        ]
        file = "_and_".join(file_parts)
        assert formatter.format_file(crits) == file

    def test_or(self) -> None:
        """Don't require all criteria to be met."""
        crits = Criteria(
            [
                VariableCriterion(
                    variable="Z",
                    time_idx=3,
                    vmin=6000,
                    vmax=9000,
                ),
                VariableCriterion(
                    variable="UV",
                    time_idx=3,
                    vmin=30,
                    vmax=None,
                ),
            ],
            require_all=False,
        )
        formatter = CriteriaFormatter()
        human_parts = [
            "Z in 6000 to 9000",
            "UV >= 30",
        ]
        human = " or ".join(human_parts)
        assert formatter.format_human(crits) == human
        file_parts = [
            "z-6000-to-9000",
            "uv-ge-30",
        ]
        file = "_or_".join(file_parts)
        assert formatter.format_file(crits) == file

    def test_times_units(self) -> None:
        """Add times and units to variables."""
        crits = Criteria(
            [
                LeaveDomainCriterion().invert(),
                VariableCriterion(
                    variable="Z",
                    time_idx=0,
                    vmin=6000,
                    vmax=9000,
                ),
                VariableCriterion(
                    variable="UV",
                    time_idx=3,
                    vmin=30,
                    vmax=None,
                ),
            ],
            require_all=True,
        )
        formatter = CriteriaFormatter(
            times=[0, 2, 4, 6],
            vars_attrs={
                "Z": {
                    "units": "m",
                    "time_units": "h",
                    "time_relative": True,
                },
                "UV": {
                    "units": "m/s",
                    "time_units": "h",
                    "time_relative": True,
                },
            },
        )
        human_parts = [
            "never leaving domain",
            "Z @ +0 h in 6000 m to 9000 m",
            "UV @ +6 h >= 30 m/s",
        ]
        human = " and ".join(human_parts)
        assert formatter.format_human(crits) == human
        file_parts = [
            "never-leaving-domain",
            "z@+0h-6000m-to-9000m",
            "uv@+6h-ge-30ms-1",
        ]
        file = "_and_".join(file_parts)
        assert formatter.format_file(crits) == file
