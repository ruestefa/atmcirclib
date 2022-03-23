"""Test ``atmcirclib.traj``."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Optional
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

# First-party
from atmcirclib.traj import TrajsDataset
from atmcirclib.typing import NDIndex_T

RAW_COORDS_D: dict[str, list[float]] = {}
RAW_DATA_D: dict[str, list[list[float]]] = {}
SCALE_FACT_D: dict[str, float] = {}
DTYPE_D: dict[str, npt.DTypeLike] = {}
ATTRS_D: dict[str, dict[str, str]] = {}
REF_DATA_D: dict[str, npt.NDArray[np.generic]]

# Test data is based on the file traj_t001914_p001.nc from the simulation
# cosmo_0.04_701x661x80/20s_explicit/2016092000/traj-pvtend.45ms-1_-4h_21_12_reduced

VNAN: float = -999.0
ATTRS: dict[str, Any] = {
    "ref_year": 2016,
    "ref_month": 9,
    "ref_day": 20,
    "ref_hour": 0,
    "ref_min": 0,
    "ref_sec": 0,
    "duration_in_sec": 36000.0,
    "pollon": 178.0,
    "pollat": 30.0,
    "output_timestep_in_sec": 2400.0,
    # -> increased from 120 by 20x (``ds.time.data[::20]``)
}

_name = "time"
ATTRS_D[_name] = {
    "standard_name": "time",
    "long_name": "time",
}
DTYPE_D[_name] = "timedelta64[ns]"
RAW_COORDS_D[_name] = [
    114820000000000,
    117120000000000,
    119520000000000,
    121920000000000,
    124320000000000,
    126720000000000,
    129120000000000,
    131520000000000,
    133920000000000,
    136320000000000,
    138720000000000,
    141120000000000,
    143520000000000,
    145920000000000,
    148320000000000,
    150720000000000,
]

_name = "longitude"
ATTRS_D[_name] = {
    "standard_name": "grid_longitude",
    "long_name": "rotated longitudes",
    "units": "degrees",
}
RAW_DATA_D[_name] = [
    [-12.960000, -14.320000, -14.780000, -2.4700000, -15.640000],
    [-12.505025, -13.850332, -14.047348, -2.1941035, -15.007464],
    [-12.038691, -13.317595, -13.217683, -1.9169528, -14.308352],
    [-11.574316, -12.745070, -12.309043, -1.6397635, -13.567076],
    [-11.104815, -12.144551, -11.320708, -1.3472217, -12.811005],
    [-10.627452, -11.525356, -10.273269, -1.0364096, -12.043587],
    [-10.141058, -10.881921, -9.2290660, -0.7263492, -11.268594],
    [-9.6433800, -10.211937, -8.2418310, -0.4493223, -10.522579],
    [-9.1215140, -9.5276681, -7.3158090, -0.2476913, -9.8334950],
    [-8.5784920, -8.8416720, -6.4403667, -999.00000, -9.2089230],
    [-8.0386700, -8.1590605, -5.6454960, -999.00000, -8.6111570],
    [-7.4971560, -7.4826694, -4.9180910, -999.00000, -8.0056210],
    [-6.9767356, -6.8211860, -4.2789884, -999.00000, -7.4002714],
    [-6.4768010, -6.1848100, -3.7289991, -999.00000, -6.7966666],
    [-5.9985560, -5.5847034, -3.2629807, -999.00000, -6.1908570],
    [-5.5685160, -5.0271990, -2.8602903, -999.00000, -5.6205254],
]

_name = "latitude"
ATTRS_D[_name] = {
    "standard_name": "grid_latitude",
    "long_name": "rotated latitudes",
    "units": "degrees",
}
RAW_DATA_D[_name] = [
    [+2.5500000, +5.0600000, +4.8100000, +3.4300000, +4.8000000],
    [+2.3678443, +4.6997237, +4.2068934, +4.1080290, +4.4511647],
    [+2.2240510, +4.3305060, +3.6630123, +4.8643710, +4.0997840],
    [+2.1329813, +3.9884380, +3.2309465, +5.6740760, +3.7602365],
    [+2.0814560, +3.6640203, +2.8823450, +6.5319320, +3.4147503],
    [+2.0524142, +3.3778608, +2.6024650, +7.4053082, +3.0629180],
    [+2.0593994, +3.1522596, +2.4003577, +8.2910230, +2.7279482],
    [+2.1091450, +2.9982774, +2.2788532, +9.1655500, +2.4274735],
    [+2.1890922, +2.9071543, +2.2233200, +9.9618360, +2.1702688],
    [+2.2708333, +2.8735678, +2.2188810, -999.00000, +1.9860597],
    [+2.3567126, +2.8928337, +2.2785673, -999.00000, +1.8982346],
    [+2.4389322, +2.9614692, +2.3749216, -999.00000, +1.8804827],
    [+2.5230250, +3.0884900, +2.5059407, -999.00000, +1.9150730],
    [+2.6045040, +3.2733364, +2.6803412, -999.00000, +2.0064213],
    [+2.6914992, +3.5170245, +2.9033399, -999.00000, +2.1465209],
    [+2.8097346, +3.8092180, +3.1882650, -999.00000, +2.3370264],
]

_name = "z"
ATTRS_D[_name] = {
    "standard_name": "height",
    "long_name": "height above mean sea level",
    "units": "m AMSL",
}
RAW_DATA_D[_name] = [
    [+2328.1001, +4935.1001, +9379.5996, +10340.500, +11840.000],
    [+2320.4250, +4978.9507, +9409.8691, +10437.542, +11902.035],
    [+2384.1274, +4946.9824, +9424.6426, +10485.778, +11873.618],
    [+2446.5947, +4999.7505, +9424.4443, +10497.885, +11818.410],
    [+2498.8008, +4912.8887, +9573.5664, +10465.488, +11795.456],
    [+2519.8657, +4950.9199, +9634.5049, +10422.081, +11710.216],
    [+2501.8445, +4927.5420, +9605.4219, +10241.846, +11535.961],
    [+2473.8311, +4769.9897, +9605.8760, +10163.859, +11466.783],
    [+2418.2976, +4697.5767, +9708.0918, +10111.673, +11549.748],
    [+2433.3423, +4640.9438, +9738.9707, -999.00000, +11666.730],
    [+2564.6218, +4715.2422, +9789.2646, -999.00000, +11780.478],
    [+2610.8276, +4745.6353, +9737.4971, -999.00000, +11762.130],
    [+2653.3411, +4819.9473, +9730.7051, -999.00000, +11688.423],
    [+2544.1484, +4843.6475, +9682.3838, -999.00000, +11670.684],
    [+2405.9883, +4848.9653, +9652.2900, -999.00000, +11644.915],
    [+2497.5054, +4834.1855, +9549.0977, -999.00000, +11686.900],
]

_name = "DPV_TOT"  # renamed from PVTEND_TOT
ATTRS_D[_name] = {
    "standard_name": "-",
    "long_name": "inst. total pv tendency",  # fixed error in name
    "units": "K m2 kg-1 s-2",
}
SCALE_FACT_D[_name] = 1e-10
RAW_DATA_D[_name] = [  # multiplied by 1e10 except -999
    [+0.0000000, +0.0000000, +0.0000000, +0.00000000, +0.0000000],
    [+0.8324232, +1.6975977, -0.5365997, +0.64774483, +1.6439659],
    [+0.6970594, +0.6700674, -1.4279434, +1.17940629, -0.3329273],
    [+0.9815146, +3.0748353, -1.8660247, -16.7464027, +0.7267463],
    [-0.2767963, +6.5477953, -0.0364937, +15.2390298, -0.0439651],
    [-1.2247152, -0.9605344, -1.7307599, -2.05600786, +7.9654178],
    [+0.9624096, -2.6638610, -1.2979512, +3.30015802, -0.2824873],
    [+0.1353499, +0.4263266, +0.0767688, +9.95089817, +4.8271322],
    [-0.7351252, -0.6380021, -4.4593548, -2.08033419, -8.7702703],
    [-1.4118857, -1.1990010, +0.7833216, -999.000000, -18.736034],
    [-3.0766410, -0.2798552, +0.3066947, -999.000000, +14.727112],
    [+2.7542398, -1.1563367, +1.7974579, -999.000000, -7.7553496],
    [-2.3529722, -1.7729704, +0.6594197, -999.000000, -26.627998],
    [+1.0978214, -0.4403861, +3.8857874, -999.000000, -36.900268],
    [-9.1229906, -0.2926828, -0.5514700, -999.000000, -14.271090],
    [+1.4542306, -2.5619907, -5.9217143, -999.000000, +10.669286],
]

_name = "DPV_TUR"  # renamed from PVTEND_TUR
ATTRS_D[_name] = {
    "standard_name": "-",
    "long_name": "inst. pv tendency due to turbulence",  # shortened name
    "units": "K m2 kg-1 s-2",
}
SCALE_FACT_D[_name] = 1e-10
RAW_DATA_D[_name] = [  # multiplied by 1e10 except -999
    [+0.0000000, +0.0000000, +0.0000000, +0.0000000, +0.0000000],
    [-0.0194929, -0.0444286, +0.0051638, -0.0628627, +0.0509059],
    [-0.0356815, -0.0104770, +0.5621650, -0.3920508, +0.0173109],
    [-0.0577381, -0.0302751, -0.1958460, +0.3707606, -0.0389508],
    [-0.0190167, -0.0768815, +0.1466828, -0.2804043, -0.0513476],
    [+0.0145035, -0.0628085, +0.0223988, +0.8200881, +0.7848811],
    [+0.0264017, -0.0455414, +0.2546478, -0.2068241, +1.5611288],
    [+0.0399352, +0.0145189, +0.0411778, -0.8076038, -1.4786950],
    [+0.0190902, +0.0102567, +0.3850102, +2.9374423, +7.3098139],
    [-0.0065218, +0.0205679, +0.3980656, -999.00000, -1.8655865],
    [-0.0145358, +0.0137055, +0.1189567, -999.00000, -2.1191535],
    [-0.0021867, +0.0005607, +0.3841197, -999.00000, -1.2541530],
    [+0.0023971, -0.0229591, -0.1084374, -999.00000, -0.3356334],
    [+0.0240532, -0.0204966, -0.1628013, -999.00000, -2.6937379],
    [-0.5047974, +0.0018002, +0.1874089, -999.00000, -0.0153870],
    [-2.0506441, -0.0049400, +0.1469335, -999.00000, -0.4097483],
]

_name = "DPV_MPHYS"  # renamed from PVTEND_MPHYS
ATTRS_D[_name] = {
    "standard_name": "-",
    "long_name": "inst. pv tendency due to microphysics",
    "units": "K m2 kg-1 s-2",
}
SCALE_FACT_D[_name] = 1e-12
RAW_DATA_D[_name] = [  # multiplied by 1e12 except -999
    [+0.0000000, +0.0000000, +0.0000000, +0.0000000, +0.0000000],
    [+0.0031990, -0.0589404, +0.1226051, -0.1815614, -0.4974059],
    [+0.0909058, -0.3249734, +0.0922187, -0.9558949, -0.0925820],
    [-0.3012517, -0.0808106, -0.1270511, -0.2441671, +0.0069390],
    [-0.0389332, +0.5614893, -0.2332214, +0.4229459, -0.1704341],
    [-0.1523829, -0.2632451, -0.2821975, -0.1653455, +2.1632785],
    [+0.1636860, -0.3256340, -0.1880787, -1.2338594, +1.4643534],
    [+0.1750009, -0.0204703, +0.4543868, +1.2972723, +1.5503745],
    [+0.1494471, -0.1054329, -0.2316762, -0.8621401, -2.0025811],
    [-0.1011912, -0.0338045, +0.0432772, -999.00000, +0.0771197],
    [+0.0872556, +0.3738496, +0.1580462, -999.00000, -2.0282650],
    [+0.1336367, +0.1698360, +0.5155500, -999.00000, +1.0323950],
    [-0.1431506, +0.1274155, +0.1592175, -999.00000, -1.7079294],
    [-0.3654358, +0.3114384, -0.4755508, -999.00000, -0.4428356],
    [+0.7403941, -0.0350316, -0.0347499, -999.00000, -0.1732392],
    [-0.0378837, -0.0644378, +0.0945929, -999.00000, -0.3563101],
]


def get_ref_arrs(
    raw_d: Union[dict[str, list[float]], dict[str, list[list[float]]]]
) -> dict[str, npt.NDArray[np.generic]]:
    """Turn raw value lists into properly scaled data arrays."""
    ref_d: dict[str, npt.NDArray[np.generic]] = {}
    arr: npt.NDArray[np.generic]
    for name, raw_data in raw_d.items():
        arr = np.array(raw_data, DTYPE_D.get(name, np.float32))
        if name in SCALE_FACT_D:
            arr = np.where(arr == VNAN, VNAN, arr * SCALE_FACT_D[name])
        ref_d[name] = arr
    return ref_d


REF_DATA_D = get_ref_arrs(RAW_DATA_D)
REF_COORDS_D = get_ref_arrs(RAW_COORDS_D)


def create_trajs_xr_dataset(
    *,
    attrs: Optional[dict[str, Any]] = None,
    coords_d: Optional[dict[str, npt.NDArray[np.generic]]] = None,
    data_d: Optional[dict[str, npt.NDArray[np.generic]]] = None,
    attrs_d: Optional[dict[str, dict[str, str]]] = None,
) -> xr.Dataset:
    """Create a mock trajs xarray dataset as read from a NetCDF file."""
    if attrs is None:
        attrs = ATTRS
    if coords_d is None:
        coords_d = REF_COORDS_D
    if data_d is None:
        data_d = REF_DATA_D
    if attrs_d is None:
        attrs_d = ATTRS_D

    if n := len(coords_d) != 1:
        raise NotImplementedError(
            f"{n} coords: " + ", ".join(map("'{}'".format, coords_d))
        )
    coord_name = next(iter(coords_d))
    dims = (coord_name, "id")

    def create_coord(name: str) -> xr.DataArray:
        """Create coordinate variable."""
        assert coords_d is not None  # mypy
        assert attrs_d is not None  # mypy
        return xr.DataArray(
            data=coords_d[name].copy(),
            coords={name: coords_d[name].copy()},
            dims=(name,),
            name=name,
            attrs=dict(attrs_d[name]),
        )

    def create_variable(name: str) -> xr.DataArray:
        """Create variable data array."""
        assert data_d is not None  # mypy
        assert attrs_d is not None  # mypy
        return xr.DataArray(
            data=data_d[name].copy(),
            dims=dims,
            name=name,
            attrs=dict(attrs_d[name]),
        )

    return xr.Dataset(
        coords={name: create_coord(name) for name in coords_d},
        data_vars={name: create_variable(name) for name in data_d},
        attrs=dict(attrs),
    )


# pylint: disable=R0201  # no-self-use
class Test_TestData:
    """Test test data."""

    def test_create_trajs_xr_dataset(self) -> None:
        """Test creation of a mock trajs dataset."""
        ds = create_trajs_xr_dataset()
        assert ds.attrs == ATTRS
        # Check time coordinate
        assert set(dict(ds.coords)) == {"time"}
        assert ds.coords["time"].attrs == ATTRS_D["time"]
        assert (ds.coords["time"].data.astype(int) == REF_COORDS_D["time"]).all()
        # Check variables
        assert set(dict(ds.variables).keys()) == set(REF_DATA_D) | set(REF_COORDS_D)
        for name, data in REF_DATA_D.items():
            assert ds.variables[name].attrs == ATTRS_D[name]
            assert np.allclose(ds.variables[name].data, data)

    def test_ref_data(self) -> None:
        """Make sure dataset contains copies of ref array."""
        name = "z"
        ds = create_trajs_xr_dataset()
        assert np.allclose(ds.variables[name].data, REF_DATA_D[name].data)
        mask = ds.variables[name].data > 3000
        assert mask.sum() > 0
        ds.variables[name].data[mask] += 100
        assert not np.allclose(ds.variables[name].data, REF_DATA_D[name].data)


# pylint: disable=R0201  # no-self-use
class Test_Init:
    """Test initialization."""

    def test_fail(self) -> None:
        """Initialize without arguments, which should fail."""
        with pytest.raises(TypeError):
            # pylint: disable=E1120  # no-value-for-parameter
            TrajsDataset()  # type: ignore  # noqa

    def test_ds(self) -> None:
        """Initalize with xarray dataset."""
        ds = create_trajs_xr_dataset()
        trajs = TrajsDataset(ds)
        assert trajs.ds == ds

    def test_config(self) -> None:
        """Initialize with changed config parameter."""
        ds = create_trajs_xr_dataset()
        trajs_ref = TrajsDataset(ds)
        trajs_exp = TrajsDataset(ds, nan=666)
        assert trajs_ref.config.nan == -999
        assert trajs_exp.config.nan == 666


class Test_GetData:
    """Test method ``get_data``."""

    def test_default(self) -> None:
        """Call with default options, whereby -999 are replaced by nans."""
        trajs = TrajsDataset(create_trajs_xr_dataset())
        for name, ref in REF_DATA_D.items():
            # Raw field contains -999
            exp = trajs.ds.variables[name].data
            assert np.allclose(exp, ref)
            # -999 replaced by nans by default
            exp = trajs.get_data(name)
            assert not np.allclose(exp, ref, equal_nan=True)
            ref = np.where(ref == VNAN, np.nan, ref)
            assert np.allclose(exp, ref, equal_nan=True)

    def test_default_explicit(self) -> None:
        """Call with explicit default values."""
        trajs = TrajsDataset(create_trajs_xr_dataset())
        for name, ref in REF_DATA_D.items():
            ref = trajs.get_data(name)
            exp = trajs.get_data(
                name=name,
                idx_time=None,
                idx_traj=None,
                replace_vnan=True,
            )
            assert np.allclose(ref, exp, equal_nan=True)

    def test_replace_vnan(self) -> None:
        """Don't replace -999 by nans."""
        trajs = TrajsDataset(create_trajs_xr_dataset())
        for name, ref in REF_DATA_D.items():
            exp = trajs.get_data(name, replace_vnan=False)
            assert np.allclose(exp, ref, equal_nan=True)

    @dc.dataclass
    class IndexingTestParams:
        """Parameters passed to ``test_indexing``."""

        idx_time: NDIndex_T = None
        idx_traj: NDIndex_T = None

    @pytest.mark.parametrize(
        "c",
        [
            IndexingTestParams(None, None),
            IndexingTestParams(0, None),
            IndexingTestParams(None, -1),
            IndexingTestParams(3, 4),
            IndexingTestParams((2, 3), 0),
            IndexingTestParams(None, slice(None, None, 3)),
            IndexingTestParams(slice(4, -1, 3), slice(None, None, -1)),
        ],
    )
    def test_indexing(self, c: IndexingTestParams) -> None:
        """Get subarrays by indexing."""
        trajs = TrajsDataset(create_trajs_xr_dataset())
        idcs: dict[str, NDIndex_T] = {}
        if c.idx_time is not None:
            idcs["idx_time"] = c.idx_time
        if c.idx_traj is not None:
            idcs["idx_traj"] = c.idx_traj
        for name, ref in REF_DATA_D.items():
            exp = trajs.get_data(name, replace_vnan=False, **idcs)
            ref = ref[c.idx_time, c.idx_traj]
            assert np.allclose(exp, ref)
