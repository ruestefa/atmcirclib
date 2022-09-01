"""Extensions of the typer command line library."""
from __future__ import annotations

# Standard library
from functools import wraps
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import overload
from typing import TypeVar
from typing import Union

# Third-party
import typer
from typing_extensions import ParamSpec

# First-party
from atmcirclib.click import pdb_wrap

P = ParamSpec("P")
P1 = ParamSpec("P1")
P2 = ParamSpec("P2")

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


def create_typer(*args: Any, **kwargs: Any) -> typer.Typer:
    """Create instance of ``typer.Typer`` with adapted defaults."""
    kwargs = {
        "add_completion": False,
        "context_settings": CONTEXT_SETTINGS,
        **kwargs,
    }
    return typer.Typer(*args, **kwargs)


def typer_option_pdb(
    default: bool = False,
    flag: str = "--pdb/--no-pdb",
    /,
    *,
    help: str = "Drop into debugger when an exception is raised.",
    is_eager: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a typer option for a ``--pdb`` option."""
    return typer.Option(default, flag, help=help, is_eager=is_eager, **kwargs)


@overload
def typer_wrap_pdb(fct: Callable[P1, T1], /) -> Callable[P1, T1]:
    ...


@overload
def typer_wrap_pdb(
    name: Optional[str] = ...,
    /,
) -> Callable[[Callable[P2, T2]], Callable[P2, T2]]:
    ...


def typer_wrap_pdb(
    fct_or_name: Optional[Union[Callable[P1, T1], str]] = None,
    /,
) -> Union[Callable[P1, T1], Callable[[Callable[P2, T2]], Callable[P2, T2]]]:
    """Decorate a typer command function to wrap it with ``pdb_wrap``.

    Whether the function is wrapped -- in which case the program drops into the
    pdb or ipdb debugger when an exception is raised -- depends on the keyword
    argument with the default name ``pdb``.

    """

    def wrap_fct(fct: Callable[P, T], name: str) -> Callable[P, T]:
        """Wrap function ``fct``."""

        @wraps(fct)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            """Wrap function ``fct``."""
            try:
                do_wrap = kwargs[name]
            except KeyError as e:
                raise ValueError(
                    f"missing '{name}' in kwargs to function '{fct.__name__}': {kwargs}"
                ) from e
            return (pdb_wrap(fct) if do_wrap else fct)(*args, **kwargs)

        return wrapped

    default_name = "pdb"
    if callable(fct_or_name):
        # Case 1: Decorator not called (``@typer_wrap_pdb``)
        return wrap_fct(fct=fct_or_name, name=default_name)
    else:
        # Case 2: Decorator called (``@typer_wrap_pdb(...)``)
        def inner(fct: Callable[P, T]) -> Callable[P, T]:
            """Decorate function ``fct``."""
            name = cast(str, default_name if fct_or_name is None else fct_or_name)
            return wrap_fct(fct=fct, name=name)

        return inner
