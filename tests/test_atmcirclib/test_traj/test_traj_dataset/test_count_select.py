"""Test ``atmcirclib.traj``."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from typing import Any
from typing import cast

# Third-party
import numpy as np
import numpy.typing as npt
import pytest

# First-party
from atmcirclib.cosmo import COSMOGridDataset
from atmcirclib.traj import BoundaryZoneCriterion
from atmcirclib.traj import Criteria
from atmcirclib.traj import LeaveDomainCriterion
from atmcirclib.traj import TrajDataset
from atmcirclib.traj import VariableCriterion

# Local
from .shared import create_cosmo_grid_dataset_ds
from .shared import TrajsDatasetDsFactory

# pylint: disable=R0201  # no-self-use

RAW_COORDS_D: dict[str, list[float]] = {}
RAW_DATA_D: dict[str, list[list[float]]] = {}
SCALE_FACT_D: dict[str, float] = {}
DTYPE_D: dict[str, npt.DTypeLike] = {}
ATTRS_D: dict[str, dict[str, str]] = {}
REF_DATA_D: dict[str, npt.NDArray[np.generic]]

RLON_LIM: tuple[float, float] = (-10.0, 0.0)
RLAT_LIM: tuple[float, float] = (-10.0, 0.0)
Z_LIM: tuple[float, float] = (0.0, 20_000.0)

# mypy 0.941 thinks arange returns signed-int array (numpy 1.22.3)
RLON: npt.NDArray[np.float_] = cast(
    npt.NDArray[np.float_],
    np.arange(RLON_LIM[0], RLON_LIM[1] + 0.1, 1.0),
)
# mypy 0.941 thinks arange returns signed-int array (numpy 1.22.3)
RLAT: npt.NDArray[np.float_] = cast(
    npt.NDArray[np.float_],
    np.arange(RLAT_LIM[0], RLAT_LIM[1] + 0.1, 1.0),
)

VNAN: float = -999.0
ATTRS: dict[str, Any] = {
    "ref_year": 2016,
    "ref_month": 9,
    "ref_day": 20,
    "ref_hour": 0,
    "ref_min": 0,
    "ref_sec": 0,
    "duration_in_sec": 36000.0,
    "pollon": 180.0,
    "pollat": 32.0,
    "output_timestep_in_sec": 3600.0,
}

_name = "time"
ATTRS_D[_name] = {
    "standard_name": "time",
    "long_name": "time",
}
DTYPE_D[_name] = "timedelta64[h]"
RAW_COORDS_D[_name] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

_name = "longitude"
ATTRS_D[_name] = {
    "standard_name": "grid_longitude",
    "long_name": "rotated longitudes",
    "units": "degrees",
}
RAW_DATA_D[_name] = [
    [-9.000, -2.000, -2.500, -5.250, -5.250, -8.000],
    [-8.000, -2.100, -2.000, -5.300, -5.300, -7.500],
    [-7.250, -2.250, -1.500, -5.350, -5.350, -7.000],
    [-6.750, -2.500, -1.000, -5.500, -5.500, -6.000],
    [-6.500, -2.650, -0.500, -5.650, -5.650, -5.000],
    [-6.250, -2.600, -0.000, -5.900, -5.900, -4.250],
    [-6.000, -2.450, -999.0, -6.300, -6.300, -3.750],
    [-6.000, -2.200, -999.0, -6.600, -6.600, -3.500],
    [-6.000, -2.000, -999.0, -7.000, -7.000, -3.250],
    [-6.000, -1.750, -999.0, -999.0, -7.500, -3.000],
]

_name = "latitude"
ATTRS_D[_name] = {
    "standard_name": "grid_latitude",
    "long_name": "rotated latitudes",
    "units": "degrees",
}
RAW_DATA_D[_name] = [
    [+1.500, -3.400, -0.500, -3.200, -3.100, +1.000],
    [+1.750, -3.600, -0.400, -3.300, -3.200, +0.900],
    [+2.000, -3.550, -0.300, -3.400, -3.300, +0.800],
    [+2.500, -3.550, -0.200, -3.500, -3.400, +0.700],
    [+3.000, -3.250, -0.100, -3.800, -3.500, +0.600],
    [+4.000, -3.000, +0.000, -4.000, -3.800, +0.500],
    [+5.000, -2.800, -999.0, -4.250, -4.000, +0.350],
    [+6.000, -2.750, -999.9, -4.500, -4.250, +0.150],
    [+7.500, -2.750, -999.0, -5.000, -4.500, -0.800],
    [+9.000, -2.800, -999.0, -999.0, -5.000, -0.600],
]

_name = "z"
ATTRS_D[_name] = {
    "standard_name": "height",
    "long_name": "height above mean sea level",
    "units": "m AMSL",
}
SCALE_FACT_D[_name] = 1e3  # km => m
RAW_DATA_D[_name] = [
    [+10.00, +3.800, +6.000, +1.000, +1.000, +2.200],
    [+10.30, +3.850, +6.100, +1.100, +1.100, +2.250],
    [+10.60, +3.800, +6.100, +1.000, +1.000, +2.200],
    [+10.60, +4.000, +6.000, +1.400, +1.400, +2.150],
    [+10.20, +4.500, +5.900, +2.000, +2.000, +2.150],
    [+9.800, +5.250, +5.900, +3.500, +3.500, +2.400],
    [+9.700, +5.500, -999.0, +5.500, +5.500, +2.850],
    [+9.900, +5.650, -999.0, +8.000, +8.000, +3.100],
    [+10.10, +5.700, -999.0, +9.500, +9.500, +3.300],
    [+10.00, +5.700, -999.0, -999.0, +9.500, +3.350],
]

_name = "U"
ATTRS_D[_name] = {
    "standard_name": "grid_eastward_wind",
    "long_name": "U-component of wind",
    "units": "m s-1",
}
RAW_DATA_D[_name] = [
    [+110.0, -11.00, +55.00, -5.500, -5.500, +55.00],
    [+96.25, -13.75, +55.00, -5.500, -5.500, +55.00],
    [+68.75, -22.00, +55.00, -11.00, -11.00, +82.50],
    [+41.25, -22.00, +55.00, -16.50, -16.50, +110.0],
    [+27.50, -5.500, +55.00, -22.00, -22.00, +96.25],
    [+27.50, +11.00, +55.00, -35.75, -35.75, +68.75],
    [+13.75, +22.00, -999.0, -38.50, -38.50, +41.25],
    [+0.000, +24.75, -999.0, -38.50, -38.50, +27.50],
    [+0.000, +24.75, -999.0, -49.50, -49.50, +27.50],
    [+0.000, +27.50, -999.0, -999.0, -55.00, +27.50],
]

_name = "V"
ATTRS_D[_name] = {
    "standard_name": "grid_northward_wind",
    "long_name": "V-component of wind",
    "units": "m s-1",
}
RAW_DATA_D[_name] = [
    [+27.50, -22.00, +11.00, -11.00, -11.00, -11.00],
    [+27.50, -8.250, +11.00, -11.00, -11.00, -11.00],
    [+41.25, +2.750, +11.00, -11.00, -11.00, -11.00],
    [+55.00, +16.50, +11.00, -22.00, -11.00, -11.00],
    [+82.50, +30.25, +11.00, -27.50, -22.00, -11.00],
    [+110.0, +24.75, +11.00, -24.75, -27.50, -13.75],
    [+110.0, +13.75, -999.0, -27.50, -24.75, -19.25],
    [+137.5, +2.750, -999.0, -41.25, -27.50, -63.25],
    [+165.0, -2.750, -999.0, -55.00, -41.25, -41.25],
    [+175.0, -5.500, -999.0, -999.0, -55.00, -22.00],
]

# UV:
#   [ 113.3,   24.5,   56.0,   12.2,   12.2,   56.0],
#   [ 100.1,   16.0,   56.0,   12.2,   12.2,   56.0],
#   [  80.1,   22.1,   56.0,   15.5,   15.5,   83.2],
#   [  68.7,   27.5,   56.0,   27.5,   19.8,  110.5],
#   [  86.9,   30.7,   56.0,   35.2,   31.1,   96.8],
#   [ 113.3,   27.0,   56.0,   43.4,   45.1,   70.1],
#   [ 110.8,   25.9,    nan,   47.3,   45.7,   45.5],
#   [ 137.5,   24.9,    nan,   56.4,   47.3,   68.9],
#   [ 165.0,   24.9,    nan,   73.9,   64.4,   49.5],
#   [ 175.0,   28.0,    nan,    nan,   77.7,   35.2],

_name = "W"
ATTRS_D[_name] = {
    "standard_name": "upward_air_velocity",
    "long_name": "vertical wind velocity",
    "units": "m s-1",
}
SCALE_FACT_D[_name] = 1 / 3.6  # km/h -> m/s
RAW_DATA_D[_name] = [
    [+0.000, +0.050, +0.100, +0.100, +0.100, +0.050],
    [+0.300, +0.000, +0.050, +0.000, +0.000, +0.000],
    [+0.150, +0.075, -0.050, +0.150, +0.150, -0.050],
    [-0.200, +0.350, -0.010, +0.500, +0.500, -0.025],
    [-0.400, +0.625, -0.050, +1.050, +1.050, +0.125],
    [-0.250, +0.500, +0.050, +1.750, +1.750, +0.350],
    [+0.050, +0.200, -999.0, +2.250, +2.250, +0.350],
    [+0.200, +0.100, -999.0, +2.000, +2.000, +0.225],
    [+0.050, +0.025, -999.0, +0.750, +0.750, +0.125],
    [+0.000, +0.000, -999.0, -999.0, +0.000, +0.050],
]

# _name = "P"
# ATTRS_D[_name] = {
#     "standard_name": "air_pressure",
#     "long_name": "pressure",
#     "units": "Pa",
# }
# # SCALE_FACT_D[_name] = 1e-2
# RAW_DATA_D[_name] = [
# ]

# _name = "T"
# ATTRS_D[_name] = {
#     "standard_name": "air_temperature",
#     "long_name": "temperature",
#     "units": "K",
# }
# # SCALE_FACT_D[_name] = 1e-12
# RAW_DATA_D[_name] = [
# ]

# _name = "QV"
# ATTRS_D[_name] = {
#     "standard_name": "specific_humidity",
#     "long_name": "specific humidity",
#     "units": "kg kg-1",
# }
# # SCALE_FACT_D[_name] = 1e-12
# RAW_DATA_D[_name] = [
# ]

# _name = "POT_VORTIC"
# ATTRS_D[_name] = {
#     "standard_name": "ertel_potential_vorticity",
#     "long_name": "potential vorticity",
#     "units": "K m2 kg-1 s-1",
# }
# # SCALE_FACT_D[_name] = 1e-12
# RAW_DATA_D[_name] = [
# ]


trajs_ds_factory = TrajsDatasetDsFactory(
    attrs=ATTRS,
    raw_coords_d=RAW_COORDS_D,
    raw_data_d=RAW_DATA_D,
    attrs_d=ATTRS_D,
    dtype_d=DTYPE_D,
    scale_fact_d=SCALE_FACT_D,
    vnan=VNAN,
)

REF_DATA_D = trajs_ds_factory.ref_data_d
REF_COORDS_D = trajs_ds_factory.ref_coords_d

GRID_DS = COSMOGridDataset(
    create_cosmo_grid_dataset_ds(
        rlon=RLON,
        rlat=RLAT,
        pole_rlon=180,
        pole_rlat=32,
        height_toa_m=Z_LIM[1],
    )
)


class Test_GetData:
    """Test some fields unavailable in the data in ``test_base``."""

    def test_uv(self) -> None:
        """Get horizontal wind speed."""
        trajs = TrajDataset(trajs_ds_factory.run())
        uv = trajs.get_data("UV")
        u_ref = np.where(REF_DATA_D["U"] == VNAN, np.nan, REF_DATA_D["U"])
        v_ref = np.where(REF_DATA_D["V"] == VNAN, np.nan, REF_DATA_D["V"])
        uv_ref = np.sqrt(u_ref**2 + v_ref**2)
        assert np.allclose(uv, uv_ref, equal_nan=True)


class Test_Count:
    """Count trajs that meet given criteria."""

    def test_incomplete(self) -> None:
        """Count trajs that leave the domain."""
        trajs = TrajDataset(trajs_ds_factory.run())
        n_incomplete = trajs.count(Criteria([LeaveDomainCriterion()]))
        n_complete = trajs.count(Criteria([LeaveDomainCriterion().invert()]))
        assert n_incomplete == 2
        assert n_complete == 4

    # Temporary test to be used to reimplement selection of boundary trajs
    # (pull domain info, incl. boundary zone coords, out of TrajDataset)
    def test_boundary(self) -> None:
        """Count trajs that reach the boundary zone."""
        trajs = TrajDataset(trajs_ds_factory.run())
        n_boundary = trajs.count(
            Criteria([BoundaryZoneCriterion(grid=GRID_DS, size_deg=1)])
        )
        n_inner = trajs.count(
            Criteria([BoundaryZoneCriterion(grid=GRID_DS, size_deg=1).invert()])
        )
        assert n_boundary == 4
        assert n_inner == 2

    @dc.dataclass
    class _TestCountConfig:
        """Configuration of ``test_z``."""

        criteria: Criteria
        n: int = -1

    @pytest.mark.parametrize(
        "cf",
        [
            _TestCountConfig(  # cf[0]
                criteria=Criteria([]),
                n=4,
            ),
            _TestCountConfig(  # cf[1]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="z",
                            time_idx=0,
                            vmin=3000,
                            vmax=None,
                        ),
                    ]
                ),
                n=2,
            ),
            _TestCountConfig(  # cf[2]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="z",
                            time_idx=6,
                            vmin=None,
                            vmax=9000,
                        ),
                    ]
                ),
                n=3,
            ),
            _TestCountConfig(  # cf[3]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="z",
                            time_idx=-3,
                            vmin=7500,
                            vmax=85000,
                        ),
                    ]
                ),
                n=2,
            ),
            _TestCountConfig(  # cf[4]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="UV",
                            time_idx=5,
                            vmin=100,
                            vmax=None,
                        ),
                    ]
                ),
                n=1,
            ),
            _TestCountConfig(  # cf[5]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="UV",
                            time_idx=3,
                            vmin=20,
                            vmax=70,
                        ),
                    ]
                ),
                n=2,
            ),
            _TestCountConfig(  # cf[6]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="z",
                            time_idx=-1,
                            vmin=8000,
                            vmax=None,
                        ),
                        VariableCriterion(
                            variable="UV",
                            time_idx=-1,
                            vmin=30,
                            vmax=None,
                        ),
                    ]
                ),
                n=2,
            ),
            _TestCountConfig(  # cf[7]
                criteria=Criteria(
                    [
                        VariableCriterion(
                            variable="z",
                            time_idx=-1,
                            vmin=8000,
                            vmax=None,
                        ),
                        VariableCriterion(
                            variable="UV",
                            time_idx=-1,
                            vmin=30,
                            vmax=None,
                        ),
                    ],
                    require_all=False,
                ),
                n=3,
            ),
        ],
    )
    def test_complete(self, cf: _TestCountConfig) -> None:
        """Count only complete trajs trajs that meet the given criteria."""
        trajs = TrajDataset(trajs_ds_factory.run()).select(
            Criteria([LeaveDomainCriterion().invert()])
        )
        n = trajs.count(criteria=cf.criteria)
        assert n == cf.n

    def test_discount(self) -> None:
        """Check method discount that inverts the criteria."""
        criteria = Criteria(
            [
                VariableCriterion(
                    variable="z",
                    time_idx=-1,
                    vmin=8000,
                    vmax=None,
                ),
                VariableCriterion(
                    variable="UV",
                    time_idx=-1,
                    vmin=30,
                    vmax=None,
                ),
            ],
        )
        in_domain_criteria = Criteria([LeaveDomainCriterion().invert()])
        trajs = TrajDataset(trajs_ds_factory.run()).select(in_domain_criteria)
        assert trajs.count(criteria) == 2
        assert trajs.discount(criteria) == 2
        assert trajs.count(criteria.derive(require_all=False)) == 3
        assert trajs.discount(criteria.derive(require_all=False)) == 1
        assert trajs.count(criteria.derive(require_all=False).invert()) == 1
        assert trajs.discount(criteria.derive(require_all=False).invert()) == 3
