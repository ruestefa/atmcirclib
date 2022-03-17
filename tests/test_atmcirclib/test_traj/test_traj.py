"""Test ``atmcirclib.traj``."""
# Third-party
import pytest

# First-party
from atmcirclib.traj import TrajsDataset


class Test_Base:
    """Test basic functionality of ``TrajsDataset``."""

    @pytest.mark.xfail(raises=TypeError)
    def test_init_fail(self) -> None:
        """Initialization w/o arguments fails."""
        TrajsDataset()  # type: ignore
