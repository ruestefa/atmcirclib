"""Test ``atmcirclib.math``."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Type
from typing import Union

# Third-party
import pytest
from pytest import approx

# First-party
from atmcirclib.math import step_ceil


class Test_StepCeil:
    """Test function ``step_ceil``."""

    @dc.dataclass
    class Params:
        """Test parameters for ``test_step_ceil``."""

        f: float
        step: float
        sol: Union[float, Type[Exception]]
        scale: bool = False

    def test_defaults(self) -> None:
        """Test default values."""
        assert step_ceil(0.2) == step_ceil(0.2, 1.0)
        assert step_ceil(0.2) != step_ceil(0.2, 0.5)
        assert step_ceil(0.02) == step_ceil(0.02, scale=False)
        # assert step_ceil(0.02) != step_ceil(0.02, scale=True)

    def run_test(self, p: Params) -> None:
        """Run test with given params."""
        if isinstance(p.sol, float):
            res = step_ceil(p.f, p.step, scale=p.scale)
            assert res == approx(p.sol)
        else:
            with pytest.raises(p.sol):
                step_ceil(p.f, p.step, scale=p.scale)

    @pytest.mark.parametrize(
        "p",
        [
            Params(0.50, 1.00, 1.00),  # [p0]
            Params(0.60, 0.50, 1.00),  # [p1]
            Params(0.40, 0.50, 0.50),  # [p2]
            Params(0.05, 1.00, 1.00),  # [p3]
            Params(5.00, 1.00, 5.00),  # [p4]
            Params(8.00, 5.20, 10.4),  # [p5]
            Params(0.30, 5.50, 5.50),  # [p6]
            Params(0.43, 0.02, 0.44),  # [p7]
            Params(0.40, -0.5, 0.50),  # [p8]
            Params(-0.4, 0.50, 0.00),  # [p9]
        ],
    )
    def test_unscaled(self, p: Params) -> None:
        """Test with unscaled steps."""
        self.run_test(p)

    @pytest.mark.parametrize(
        "p",
        [
            Params(0.05, 0.20, NotImplementedError, scale=True),  # [p0]
        ],
    )
    def test_scaled(self, p: Params) -> None:
        """Test with scaled steps."""
        self.run_test(p)
