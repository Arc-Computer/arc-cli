"""Arc CLI console utilities for professional output formatting."""

from rich.console import Console
from rich.theme import Theme
from rich.text import Text
from typing import Any

# Professional color palette (no emojis per enterprise standards)
ARC_THEME = Theme({
    "primary": "bright_blue",      # Arc branding, headers
    "success": "bright_green",     # Improvements, positive metrics
    "warning": "bright_yellow",    # Attention items
    "error": "bright_red",         # Failures, critical issues
    "info": "bright_cyan",         # Statistical data, tips
    "muted": "bright_black",       # Secondary text
    "highlight": "bold bright_white",  # Important values
})


class ArcConsole(Console):
    """Arc-styled console for professional CLI output."""
    
    def __init__(self, **kwargs):
        super().__init__(theme=ARC_THEME, **kwargs)
    
    def print_header(self, text: str) -> None:
        """Print a styled header."""
        self.print()
        self.print(f"[primary]{text}[/primary]")
        self.print("─" * len(text), style="muted")
        self.print()
    
    def print_metric(self, label: str, value: Any, style: str = "info") -> None:
        """Print a metric with label and value."""
        self.print(f"[muted]{label}:[/muted] [{style}]{value}[/{style}]")
    
    def print_cost(self, label: str, amount: float) -> None:
        """Print cost information."""
        self.print(f"[muted]{label}:[/muted] [info]${amount:.4f}[/info]")
    
    def print_progress_update(self, message: str) -> None:
        """Print a progress update message."""
        self.print(f"[muted]├──[/muted] {message}")


def format_error(message: str) -> Text:
    """Format an error message."""
    return Text(f"✗ {message}", style="error")


def format_success(message: str) -> Text:
    """Format a success message."""
    return Text(f"✓ {message}", style="success")


def format_warning(message: str) -> Text:
    """Format a warning message."""
    return Text(f"[WARNING] {message}", style="warning")