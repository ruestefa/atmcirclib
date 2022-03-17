"""Geometric and geographic utilities."""
from __future__ import annotations

# Standard library
from collections.abc import Sequence
from typing import NamedTuple
from typing import Optional
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt


class BoundingBox(NamedTuple):
    """A rectangular bounding box."""

    llx: float
    urx: float
    lly: float
    ury: float

    def get_center(self) -> tuple[float, float]:
        """Get the center lon/lat coordinates."""
        return (0.5 * (self.llx + self.urx), 0.5 * (self.lly + self.ury))

    def get_width(self) -> float:
        """Get width."""
        return self.urx - self.llx

    def get_height(self) -> float:
        """Get height."""
        return self.ury - self.lly

    def get_aspect(self) -> float:
        """Get aspect ratio."""
        return self.get_width() / self.get_height()

    def get_xlim(self) -> tuple[float, float]:
        """Get limits in x direction."""
        return (self.llx, self.urx)

    def get_ylim(self) -> tuple[float, float]:
        """Get limits in y direction."""
        return (self.lly, self.ury)

    def shrink(self, bnd: float) -> BoundingBox:
        """Return a copy shrunk by ``bnd`` on all four sides; stop at center."""
        llx = self.llx + bnd
        lly = self.lly + bnd
        urx = self.urx - bnd
        ury = self.ury - bnd
        if llx > urx:
            llx = urx = 0.5 * (llx + urx)
        if lly > ury:
            lly = ury = 0.5 * (lly + ury)
        return type(self)(llx=llx, urx=urx, lly=lly, ury=ury)

    def swapaxes(self) -> BoundingBox:
        """Return a copy with swapped x and y axes."""
        return type(self)(llx=self.lly, urx=self.ury, lly=self.llx, ury=self.urx)

    def derive(
        self,
        *,
        llx: Optional[float] = None,
        urx: Optional[float] = None,
        lly: Optional[float] = None,
        ury: Optional[float] = None,
    ) -> BoundingBox:
        """Derive a copy with adapted parameters."""
        return type(self)(
            llx=self.llx if llx is None else llx,
            urx=self.urx if urx is None else urx,
            lly=self.lly if lly is None else lly,
            ury=self.ury if ury is None else ury,
        )

    @classmethod
    def from_coords(
        cls,
        xs: Union[Sequence[float], npt.NDArray[np.float_]],
        ys: Union[Sequence[float], npt.NDArray[np.float_]],
    ) -> BoundingBox:
        """Create a new instance from coordinate arrays."""
        return cls(llx=xs[0], urx=xs[-1], lly=ys[0], ury=ys[-1])
