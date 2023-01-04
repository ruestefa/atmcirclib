"""Top-level package ``atmcirclib``."""
# Standard library
import importlib.metadata

# Local
from . import click
from . import cosmo
from . import deriv
from . import fortran
from . import geo
from . import icon
from . import intp
from . import math
from . import plot
from . import simulations
from . import traj
from . import typer
from . import typing
from . import utils

__author__ = "Stefan Ruedisuehli"
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = importlib.metadata.version(__package__)

del importlib
