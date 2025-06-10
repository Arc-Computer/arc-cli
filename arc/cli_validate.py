"""CLI command for validating agent configurations."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import yaml

from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer

console = Console()


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--show-normalized', '-n', is_flag=True, help='Show normalized configuration')
@click.option('--show-capabilities', '-c', is_flag=True, help='Show extracted capabilities')
def validate(config_path: str, show_normalized: bool, show_capabilities: bool):
    """Validate an agent configuration file.
    
    This command checks that your agent configuration is valid and shows
    how Arc will interpret it.
    """
    config_path = Path(config_path)
    
    console.print(f"\n[bright_blue]Validating agent configuration:[/bright_blue] {config_path.name}\n")
    
    # Parse configuration
    parser = AgentConfigParser()
    try:
        parsed_config = parser.parse(config_path)
        console.print("[bright_green]✓[/bright_green] Configuration parsed successfully")
    except Exception as e:
        console.print(f"[bright_red]✗ Parsing failed:[/bright_red] {e}")
        return
    
    # Show warnings if any
    warnings = parser.get_warnings()
    if warnings:
        console.print("\n[bright_yellow]Warnings:[/bright_yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
    
    # Normalize configuration
    normalizer = ConfigNormalizer()
    normalized_config = normalizer.normalize(parsed_config)
    
    # Show enhancements applied
    enhancements = normalizer.get_enhancements_applied()
    if enhancements:
        console.print("\n[bright_cyan]Enhancements applied:[/bright_cyan]")
        for enhancement in enhancements:
            console.print(f"  • {enhancement}")
    
    # Validate normalized config
    if normalizer.validate_normalized_config(normalized_config):
        console.print("\n[bright_green]✓[/bright_green] Configuration is valid and ready for Arc")
    else:
        console.print("\n[bright_red]✗[/bright_red] Normalized configuration validation failed")
        return
    
    # Extract capabilities
    capabilities = parser.extract_capabilities(parsed_config)
    
    # Create Arc profile
    profile = normalizer.create_arc_profile(normalized_config, capabilities)
    
    # Display summary table
    console.print("\n[bright_blue]Configuration Summary:[/bright_blue]")
    
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bright_cyan")
    table.add_column("Value")
    
    table.add_row("Model", normalized_config["model"])
    table.add_row("Temperature", str(normalized_config["temperature"]))
    table.add_row("Tools", ", ".join(normalized_config["tools"]))
    table.add_row("Complexity", capabilities["complexity_level"])
    table.add_row("Domains", ", ".join(capabilities["domains"]))
    
    console.print(table)
    
    # Show test parameters
    console.print("\n[bright_blue]Recommended Test Parameters:[/bright_blue]")
    test_params = profile["test_parameters"]
    
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="bright_cyan")
    params_table.add_column("Value")
    
    params_table.add_row("Scenario Count", str(test_params["scenario_count"]))
    params_table.add_row("Timeout (seconds)", str(test_params["timeout_seconds"]))
    params_table.add_row("Parallel Workers", str(test_params["max_workers"]))
    
    if test_params.get("require_deterministic"):
        params_table.add_row("Deterministic Mode", "Yes")
    if test_params.get("mock_external_calls"):
        params_table.add_row("Mock External APIs", "Yes")
    
    console.print(params_table)
    
    # Show optimization targets
    if profile["optimization_targets"]:
        console.print("\n[bright_blue]Optimization Opportunities:[/bright_blue]")
        for target in profile["optimization_targets"]:
            console.print(f"  • {target.replace('_', ' ').title()}")
    
    # Show normalized config if requested
    if show_normalized:
        console.print("\n[bright_blue]Normalized Configuration:[/bright_blue]")
        # Remove internal fields for display
        display_config = {k: v for k, v in normalized_config.items() 
                         if k not in ["arc_metadata", "tools_normalized"]}
        yaml_str = yaml.dump(display_config, default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, border_style="bright_black"))
    
    # Show capabilities if requested
    if show_capabilities:
        console.print("\n[bright_blue]Extracted Capabilities:[/bright_blue]")
        
        # Tool categories
        if capabilities["tool_categories"]:
            console.print("\n  [bright_cyan]Tool Categories:[/bright_cyan]")
            for category, tools in capabilities["tool_categories"].items():
                console.print(f"    • {category}: {', '.join(tools)}")
        
        # Behavioral traits
        if capabilities["behavioral_traits"]:
            console.print("\n  [bright_cyan]Behavioral Traits:[/bright_cyan]")
            for trait in capabilities["behavioral_traits"]:
                console.print(f"    • {trait.replace('_', ' ').title()}")
    
    console.print("\n[bright_green]✓[/bright_green] Validation complete\n")


if __name__ == '__main__':
    validate()