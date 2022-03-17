"""Test ``atmcirclib.traj``."""
# Third-party
import pytest

# First-party
from atmcirclib.traj import TrajsDataset


# pylint: disable=R0201  # no-self-use
class Test_Base:
    """Test basic functionality of ``TrajsDataset``."""

    @pytest.mark.xfail(raises=TypeError)
    def test_init_fail(self) -> None:
        """Initialization w/o arguments fails."""
        # pylint: disable=E1120  # no-value-for-parameter
        TrajsDataset()  # type: ignore  # noqa
