"""Arc validate command - validate agent configuration."""

import click
from pathlib import Path

from arc.cli.utils import ArcConsole, format_error, format_success
from arc.cli_validate import validate as validate_config

console = ArcConsole()


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--show-normalized', '-n', is_flag=True, help='Show normalized configuration')
@click.option('--show-capabilities', '-c', is_flag=True, help='Show extracted capabilities')
def validate(config_path: str, show_normalized: bool, show_capabilities: bool):
    """Validate an agent configuration file.
    
    This command checks that your agent configuration is valid and shows
    how Arc will interpret it.
    
    Example:
        arc validate finance_agent.yaml
        arc validate agent.yaml --show-normalized
    """
    # Delegate to the existing validate implementation
    ctx = click.get_current_context()
    ctx.invoke(
        validate_config,
        config_path=config_path,
        show_normalized=show_normalized,
        show_capabilities=show_capabilities
    )