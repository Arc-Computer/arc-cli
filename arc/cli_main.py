"""Developer-first CLI for Arc: Proactive Capability Assurance.

Simplified commands for Applied ML Engineers to test and improve AI agents
before production deployment.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import sys

import click
from dotenv import load_dotenv
from rich.panel import Panel

from arc.cli.commands import analyze, diff, recommend, run, status, validate
from arc.cli.utils import ArcConsole

# Initialize Rich console with Arc styling
console = ArcConsole()


def _get_version() -> str:
    """Return the installed version of arc-cli."""
    try:
        return _metadata.version("arc-cli")
    except _metadata.PackageNotFoundError:
        return "0.1.0-dev"


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120
    },
    invoke_without_command=True
)
@click.version_option(_get_version(), message="Arc CLI v%(version)s")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Arc: Proactive Capability Assurance for AI Agents.

    Test and improve your AI agents BEFORE production deployment.

    \b
    Basic workflow:
      arc run agent.yaml      # Test agent with generated scenarios
      arc analyze            # Analyze failures from last run
      arc recommend          # Get improvement recommendations
      arc diff v1.yaml v2.yaml  # Compare agent versions

    \b
    Examples:
      arc run finance_agent.yaml
      arc analyze --run abc123
      arc recommend --json
      arc status
    """
    # Load environment variables from .env file, but don't override existing ones
    load_dotenv(override=False)

    if ctx.invoked_subcommand is None:
        # Show welcome message when no command is provided
        console.print()
        console.print(
            Panel.fit(
                "[bright_blue]Arc: Proactive Capability Assurance[/bright_blue]\n\n"
                "Test and improve your AI agents BEFORE production deployment.\n\n"
                "Get started: [bright_cyan]arc run your_agent.yaml[/bright_cyan]",
                border_style="bright_blue"
            )
        )
        console.print()
        console.print("Run [bright_cyan]arc --help[/bright_cyan] for available commands.")
        console.print()


# Register commands
main.add_command(run.run)
main.add_command(analyze.analyze)
main.add_command(recommend.recommend)
main.add_command(diff.diff)
main.add_command(validate.validate)
main.add_command(status.status)


if __name__ == "__main__":
    sys.exit(main())

