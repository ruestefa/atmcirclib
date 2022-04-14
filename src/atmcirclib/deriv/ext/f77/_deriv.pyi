"""Interfaces of routines in extension module ``_deriv_f77``."""

# Third-party
import numpy as np
import numpy.typing as npt

# Custom types
FloatArray3D_T = npt.NDArray[np.float_]

def aura(
    gri: FloatArray3D_T,
    dir: int,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> FloatArray3D_T:
    ...

def deriv(
    field: FloatArray3D_T,
    direction: str,
    vert: FloatArray3D_T,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> FloatArray3D_T:
    ...

def div(
    u3: FloatArray3D_T,
    v3: FloatArray3D_T,
    vert: FloatArray3D_T,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> FloatArray3D_T:
    ...

def grad(
    scalar: FloatArray3D_T,
    vert: FloatArray3D_T,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> tuple[FloatArray3D_T, FloatArray3D_T]:
    ...

def gridok(
    vert: FloatArray3D_T,
    vertcoord: str,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> None:
    ...

def rot(
    u3: FloatArray3D_T,
    v3: FloatArray3D_T,
    vert: FloatArray3D_T,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    mdv: float,
    nx: int = ...,
    ny: int = ...,
    nz: int = ...,
) -> FloatArray3D_T:
    ...
