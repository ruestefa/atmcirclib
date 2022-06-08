"""Test class ``atmcirclib.geo.BoundingBox``."""
from __future__ import annotations

# Third-party
import numpy as np
import pytest

# First-party
from atmcirclib.geo import BoundingBox

# pylint: disable=R0201  # no-self-use


class Test_Init:
    """Test initialization."""

    def test_noargs_fail(self) -> None:
        """Creating a new instance w/o arguments fails."""
        with pytest.raises(TypeError):
            BoundingBox()  # type: ignore

    def test_pos(self) -> None:
        """Use only positional arguments."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        assert bbx.llx == llx
        assert bbx.urx == urx
        assert bbx.lly == lly
        assert bbx.ury == ury

    def test_kw(self) -> None:
        """Use keyword arguments."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(
            lly=lly,
            llx=llx,
            ury=ury,
            urx=urx,
        )
        assert bbx.llx == llx
        assert bbx.urx == urx
        assert bbx.lly == lly
        assert bbx.ury == ury


class Test_Derive:
    """Test method ``derive`` to derive a new instance."""

    def test_equal(self) -> None:
        """Derive a new instance with the same coordinates."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx1 = BoundingBox(llx, urx, lly, ury)
        bbx2 = bbx1.derive()
        assert bbx1 is not bbx2
        assert bbx1 == bbx2

    def test_change(self) -> None:
        """Derive a new instance with changed coordinates."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx1 = BoundingBox(llx, urx, lly, ury)
        bbx2 = bbx1.derive(llx=-32, ury=12)
        assert bbx1 is not bbx2
        assert bbx1 != bbx2
        assert bbx2.llx == -32
        assert bbx2.urx == urx
        assert bbx2.lly == lly
        assert bbx2.ury == 12


class Test_FromCoords:
    """Test class method ``from_coords`` to create an instance from coords."""

    def test(self) -> None:
        """Create an instance from coordinate arrays."""
        xs = np.arange(-20, 5.1, 2)
        ys = np.arange(30, 45.1, 2)
        bbx = BoundingBox.from_coords(xs, ys)
        assert bbx.llx == xs[0]
        assert bbx.urx == xs[-1]
        assert bbx.lly == ys[0]
        assert bbx.ury == ys[-1]


class Test_Get:
    """Test methods ``get_*`` that return basic bbox properties."""

    def test_center(self) -> None:
        """Get center of box."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        cx, cy = bbx.get_center()
        assert np.isclose(cx, 0.5 * (llx + urx))
        assert np.isclose(cy, 0.5 * (lly + ury))

    def test_width(self) -> None:
        """Get width of box."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        w = bbx.get_width()
        assert np.isclose(w, urx - llx)

    def test_height(self) -> None:
        """Get height of box."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        h = bbx.get_height()
        assert np.isclose(h, ury - lly)

    def test_aspect(self) -> None:
        """Get aspect ratio of box."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        a = bbx.get_aspect()
        assert np.isclose(a, (urx - llx) / (ury - lly))

    def test_xlim(self) -> None:
        """Get limits of box in x-direction."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        xl = bbx.get_xlim()
        assert np.allclose(xl, (llx, urx))

    def test_ylim(self) -> None:
        """Get limits of box in y-direction."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        yl = bbx.get_ylim()
        assert np.allclose(yl, (lly, ury))


class Test_Shrink:
    """Test method ``shrink`` to reduce the size of the bbox."""

    def test_copy(self) -> None:
        """Check that the original instance is not affected by shrinking."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx1 = BoundingBox(llx, urx, lly, ury)
        bbx2 = bbx1.shrink(0)
        assert bbx1 is not bbx2
        bbx1.shrink(1)
        assert bbx1 == bbx2

    def test_scalar(self) -> None:
        """Shrink by the same amount in all directions."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        bbx = bbx.shrink(1)
        assert bbx.llx == llx + 1
        assert bbx.urx == urx - 1
        assert bbx.lly == lly + 1
        assert bbx.ury == ury - 1

    def test_tuple(self) -> None:
        """Shrink by different amounts in the x- and y-direction."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx = BoundingBox(llx, urx, lly, ury)
        bbx = bbx.shrink((2, 3))
        assert bbx.llx == llx + 2
        assert bbx.urx == urx - 2
        assert bbx.lly == lly + 3
        assert bbx.ury == ury - 3

    def test_overshrink(self) -> None:
        """Shrinking stops at the center of the box if it becomes too small."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx1 = BoundingBox(llx, urx, lly, ury)
        bbx2 = bbx1.shrink(100)
        assert bbx2.llx == bbx1.get_center()[0]
        assert bbx2.urx == bbx1.get_center()[0]
        assert bbx2.lly == bbx1.get_center()[1]
        assert bbx2.ury == bbx1.get_center()[1]

    def test_grow(self) -> None:
        """Grow in one direction (and overshrink in the other)."""
        (llx, urx, lly, ury) = (-46, 2, -15, 9)
        bbx1 = BoundingBox(llx, urx, lly, ury)
        bbx2 = bbx1.shrink((-5, 100))
        assert bbx2.llx == llx - 5
        assert bbx2.urx == urx + 5
        assert bbx2.lly == bbx1.get_center()[1]
        assert bbx2.ury == bbx1.get_center()[1]
