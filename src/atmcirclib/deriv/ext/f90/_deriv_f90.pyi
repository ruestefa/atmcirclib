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

def deformation(
    u: FloatArray3D_T,
    v: FloatArray3D_T,
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

def deriv_time(
    anew: FloatArray3D_T,
    anow: FloatArray3D_T,
    aold: FloatArray3D_T,
    pnew: FloatArray3D_T,
    pnow: FloatArray3D_T,
    pold: FloatArray3D_T,
    dt: float,
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

def gradmat(
    a1: FloatArray3D_T,
    a2: FloatArray3D_T,
    a3: FloatArray3D_T,
    b1: FloatArray3D_T,
    b2: FloatArray3D_T,
    b3: FloatArray3D_T,
    comp: str,
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

def gradvec(
    a: FloatArray3D_T,
    b: FloatArray3D_T,
    c: FloatArray3D_T,
    dir: str,
    comp: str,
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

def hadv(
    u3: FloatArray3D_T,
    v3: FloatArray3D_T,
    f3: FloatArray3D_T,
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
