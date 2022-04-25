"""Extensions of the click command line library."""
from __future__ import annotations

# Standard library
import sys
from typing import Any
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Union

# Third-party
import click

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

    # command_class = Command

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
