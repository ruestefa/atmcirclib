"""Extensions of the click command line library."""
from __future__ import annotations

# Standard library
import logging
import sys
import traceback
from functools import wraps
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Type
from typing import Union

# Third-party
import click
from click import Context
from click import Option

CONTEXT_SETTINGS = {
    "show_default": True,
    "help_option_names": ["-h", "--help"],
}


class Command(click.Command):
    """Custom click command."""

    def main(self, *args: Any, **kwargs: Any) -> Union[Any, NoReturn]:
        """Run the command, and in case of an error, provide help message."""
        try:
            return super().main(*args, standalone_mode=False, **kwargs)  # type: ignore
        except click.UsageError as exc:
            # Alternatively, only catch more specific click.MissingParameter
            echo_error_help(exc, self)
            return None


class Group(click.Group):
    """Custom click command group."""

    def invoke(self, ctx: click.Context) -> None:
        """Run the command, and in case of an error, provide help message."""
        try:
            super().invoke(ctx)
        except click.UsageError as exc:
            # Alternatively, only catch more specific click.MissingParameter
            cmd: click.Command
            if not ctx.invoked_subcommand:
                cmd = self
            else:
                cmd = cast(click.Command, self.get_command(ctx, ctx.invoked_subcommand))
                ctx = click.Context(cmd, info_name=cmd.name, parent=ctx)
            echo_error_help(exc, cmd, ctx)


def echo_error_help(
    exc: click.UsageError, cmd: click.Command, ctx: Optional[click.Context] = None
) -> None:
    """Print error message from exception, followed by the command's help.

    Originally adapted from https://stackoverflow.com/a/50976902/4419816.

    """
    exc.ctx = None
    exc.show(file=sys.stdout)
    click.echo()
    try:
        if ctx is None:
            cmd(["--help"])
        else:
            click.echo(cmd.get_help(ctx))
    except SystemExit:
        sys.exit(exc.exit_code)


def click_error(
    ctx: Context,
    msg: str,
    exception: Type[Exception] = Exception,
    echo_prefix: str = "Error: ",
) -> None:
    """Print an error message and exit, or raise an exception with traceback."""
    if ctx.obj["raise"]:
        raise exception(msg)
    else:
        click_exit(ctx, f"{echo_prefix}{msg}", stat=1)


def click_exit(ctx: Context, msg: str, stat: int = 0) -> None:
    """Exit with a message."""
    click.echo(msg, file=(sys.stdout if stat == 0 else sys.stderr))
    ctx.exit(stat)


def click_set_ctx_obj(ctx: click.Context, param: click.Option, value: Any) -> None:
    """Set the parameter in click context object.

    Pass as ``callback`` argument to a ``click.option(...)`` decorator.

    Args:
        ctx: Click context.

        param: Click option.

        value: Value of ``param``.

    Example:
        Add option to click context object::

            >>> @click.option(
            ...     "--foo/--no-foo",
            ...     callback=click_set_ctx_obj,
            ...     is_eager=True,
            ...     expose_value=False,
            ... )
            >>> @click.pass_context
            >>> def cli(ctx: click.Context) -> None:
            ...     print(f"{ctx.obj['foo']=}")

    """
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj[param.name] = value


def click_pdb_wrap_command(
    arg: str = "pdb",
    *,
    pass_context: bool = False,
) -> Callable[..., Any]:
    """Decorate a click command to drop into pdb when exception is raised.

    The command is wrapped in ``pdb_wrap`` depending on the value of ``arg`` in
    the click context object, which must be passed by preceding the decorator
    with ``@click.pass_context`` (see example below).

    Args:
        arg (optional): Name of argument in the click context object; if it is
            present and true, the command function is wrapped in ``pdb_wrap``.

        pass_context (optional): Pass the click context object on to the command
            function.

    Example:
        Wrap click command depending on global click option::

            >>> @click.group()
            >>> @click.option(
            ...     "--pdb/--no-pdb",
            ...     "drop_into_pdb",
            ...     callback=click_set_ctx_obj,
            ...     is_eager=True,
            ...     expose_value=False,
            ... )
            >>> def cli():
            ...     return 0
            >>> @cli.command()
            >>> @click.pass_context
            >>> @click_pdb_wrap_command("drop_into_pdb", pass_context=True)
            >>> def say_hi(ctx):
            ...     print("hi")

    """

    def decorator(fct: Callable[..., Any]) -> Callable[..., Any]:
        """Create the actual decorator."""

        @wraps(fct)
        def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
            """Wrap the click command function."""
            fct_ = pdb_wrap(fct) if ctx.obj.get(arg) else fct
            if pass_context:
                return fct_(ctx, *args, **kwargs)
            return fct_(*args, **kwargs)

        return wrapper

    return decorator


def click_set_raise(ctx: Context, param: Option, value: Any) -> None:
    """Set argument ``"raise"`` in click options."""
    # pylint: disable=W0613  # unused-argument (param)
    if ctx.obj is None:
        ctx.obj = {}
    if value is None:
        if "raise" not in ctx.obj:
            ctx.obj["raise"] = False
    else:
        ctx.obj["raise"] = value


def click_set_verbosity(ctx: Context, param: Option, value: Any) -> None:
    """Set argument ``"verbosity"`` in click options."""
    # pylint: disable=W0613  # unused-argument (ctx, param)
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["verbosity"] = value
    set_log_level(value)


def pdb_wrap_callback(fct: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapp click callback functions to conditionally drop into ipdb."""

    @wraps(fct)
    def wrapper(ctx: Context, param: Option, value: Any) -> Any:
        """Drop into ``ipdb`` session if ``fct`` call raises an exception."""
        fct_loc = pdb_wrap(fct) if (ctx.obj or {}).get("pdb") else fct
        return fct_loc(ctx, param, value)

    return wrapper


def pdb_wrap(fct: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a function to drop into ipdb if an exception is raised."""

    @wraps(fct)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Drop into ``ipdb`` session if ``fct`` call raises an exception."""
        try:
            return fct(*args, **kwargs)
        except Exception as e:  # pylint: disable=W0703  # broad-except
            if isinstance(e, click.exceptions.Exit):
                if e.exit_code == 0:  # pylint: disable=E1101  # no-member
                    sys.exit(0)
            pdb = __import__("ipdb")  # trick pre-commit hook "debug-statements"
            traceback.print_exc()
            click.echo()
            pdb.post_mortem()
            sys.exit(1)

    return wrapper


def set_log_level(verbosity: int) -> None:
    """Set logging level based on verbosity value."""
    if verbosity <= 0:
        logging.getLogger().setLevel(logging.INFO)
    elif verbosity == 1:
        # mypy v0.942 error: 'Module has no attribute "VERBOSE"' (python v3.9.0)
        logging.getLogger().setLevel(logging.VERBOSE)  # type: ignore
    elif verbosity >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
