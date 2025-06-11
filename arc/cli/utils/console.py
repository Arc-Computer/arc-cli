"""Arc CLI console utilities for professional output formatting."""

from rich.console import Console
from rich.theme import Theme
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from typing import Any, Dict, List

from arc.cli.design_standards import (
    COLORS, LAYOUT, SYMBOLS, ASSUMPTION_STYLES, 
    FUNNEL_STYLES, STATISTICAL_STYLES, STREAMING_CONFIG
)
from arc.cli.message_templates import (
    format_percentage, format_currency,
    format_confidence_interval, format_p_value
)

# Enterprise theme using design standards
ARC_THEME = Theme(COLORS)


class ArcConsole(Console):
    """Arc-styled console for professional CLI output with streaming support."""
    
    def __init__(self, **kwargs):
        super().__init__(theme=ARC_THEME, width=LAYOUT['terminal_width'], **kwargs)
        self._live_session = None
        self._streaming_data = []
    
    def print_header(self, text: str) -> None:
        """Print a styled header."""
        self.print()
        self.print(f"[primary]{text}[/primary]")
        self.print("─" * len(text), style="muted")
        self.print()
    
    def print_metric(self, label: str, value: Any, style: str = "info") -> None:
        """Print a metric with label and value."""
        self.print(f"[muted]{label}:[/muted] [{style}]{value}[/{style}]")
    
    def print_cost(self, amount: float) -> None:
        """Print cost information."""
        formatted_cost = format_currency(amount)
        self.print(f"[info]{formatted_cost}[/info]")
    
    def print_progress_update(self, message: str) -> None:
        """Print a progress update message."""
        self.print(f"[muted]{SYMBOLS['bullet']}[/muted] {message}")
    
    # Streaming-ready methods for real-time updates
    def start_live_session(self, auto_refresh: bool = True) -> Live:
        """Initialize live display area for streaming content."""
        if self._live_session is None:
            self._live_session = Live(
                auto_refresh=auto_refresh, 
                refresh_per_second=1.0/STREAMING_CONFIG['refresh_rate']
            )
        return self._live_session
    
    def update_progress_stream(self, data: Dict[str, Any]) -> None:
        """Update progress without clearing screen."""
        if self._live_session:
            # Update live display with new progress data
            self._live_session.update(self._render_streaming_progress(data))
    
    def append_result_stream(self, result: Dict[str, Any]) -> None:
        """Add new results to live feed."""
        self._streaming_data.append(result)
        if self._live_session:
            self._live_session.update(self._render_streaming_results())
    
    def _render_streaming_progress(self, data: Dict[str, Any]) -> Panel:
        """Render progress data for live display."""
        progress_text = f"Progress: {data.get('completed', 0)}/{data.get('total', 0)}"
        if 'cost' in data:
            progress_text += f" | Cost: {format_currency(data['cost'])}"
        return Panel(progress_text, title="Live Progress", border_style="primary")
    
    def _render_streaming_results(self) -> Table:
        """Render streaming results table."""
        table = Table(title="Real-time Results", show_header=True)
        table.add_column("Scenario", style="info")
        table.add_column("Status", style="success")
        table.add_column("Score", justify="right")
        
        # Show last N results to prevent overwhelming display
        recent_results = self._streaming_data[-10:] if len(self._streaming_data) > 10 else self._streaming_data
        
        for result in recent_results:
            status = f"{SYMBOLS['pass']} PASS" if result.get('success') else f"{SYMBOLS['fail']} FAIL"
            table.add_row(
                result.get('scenario_id', 'N/A'),
                status,
                f"{result.get('score', 0):.1f}%"
            )
        
        return table
    
    # Assumption highlighting methods
    def format_assumption(self, assumption_type: str, details: str, impact: str = None) -> Text:
        """Format assumption violation with enterprise highlighting."""
        text = Text()
        text.append("ASSUMPTION VIOLATED: ", style=ASSUMPTION_STYLES['violation_header'])
        text.append(f"{assumption_type} ", style=ASSUMPTION_STYLES['violation_detail'])
        text.append(details, style=ASSUMPTION_STYLES['violation_detail'])
        
        if impact:
            text.append(f" (Impact: {impact})", style=ASSUMPTION_STYLES['violation_impact'])
        
        return text
    
    def format_funnel(self, steps: List[Dict[str, Any]]) -> Table:
        """Format capability funnel for decomposition display."""
        table = Table(title="Capability Funnel Analysis", show_header=True)
        table.add_column("Step", style=FUNNEL_STYLES['step_header'], width=LAYOUT['funnel_step_width'])
        table.add_column("Success Rate", justify="right", style=FUNNEL_STYLES['step_success'])
        table.add_column("Failures", justify="right", style=FUNNEL_STYLES['step_failure'])
        table.add_column("Impact", style=FUNNEL_STYLES['step_rate'])
        
        for i, step in enumerate(steps):
            success_rate = step.get('success_rate', 0)
            failure_count = step.get('failures', 0)
            
            # Highlight bottlenecks
            step_style = FUNNEL_STYLES['bottleneck'] if step.get('is_bottleneck') else FUNNEL_STYLES['step_header']
            
            table.add_row(
                f"{i+1}. {step['name']}",
                format_percentage(success_rate),
                str(failure_count),
                step.get('impact', 'Normal'),
                style=step_style
            )
        
        return table
    
    def format_statistical(self, test_name: str, p_value: float, effect_size: float = None, 
                          confidence_interval: tuple = None, sample_size: int = None) -> Panel:
        """Format statistical results with enterprise credibility."""
        content = []
        
        # P-value with confidence styling
        if p_value < 0.01:
            p_style = STATISTICAL_STYLES['confidence_high']
            confidence_text = "HIGH CONFIDENCE"
        elif p_value < 0.05:
            p_style = STATISTICAL_STYLES['confidence_medium'] 
            confidence_text = "MEDIUM CONFIDENCE"
        else:
            p_style = STATISTICAL_STYLES['confidence_low']
            confidence_text = "LOW CONFIDENCE"
        
        content.append(f"{confidence_text}: {format_p_value(p_value)}")
        
        if effect_size is not None:
            content.append(f"Effect Size: d = {effect_size:.3f}")
        
        if confidence_interval:
            ci_text = format_confidence_interval(confidence_interval[0], 
                                                confidence_interval[1] - confidence_interval[0])
            content.append(f"95% CI: {ci_text}")
        
        if sample_size:
            content.append(f"Sample Size: n = {sample_size}")
        
        return Panel(
            "\n".join(content),
            title=f"Statistical Validation: {test_name}",
            border_style=p_style
        )


def format_error(message: str) -> Text:
    """Format an error message with allowed symbols only."""
    return Text(f"{SYMBOLS['fail']} {message}", style="error")


def format_success(message: str) -> Text:
    """Format a success message with allowed symbols only."""
    return Text(f"{SYMBOLS['pass']} {message}", style="success")


def format_warning(message: str) -> Text:
    """Format a warning message with professional text."""
    return Text(f"{SYMBOLS['warning_text']} {message}", style="warning")


def format_info(message: str) -> Text:
    """Format an info message with professional styling."""
    return Text(f"{SYMBOLS['info_text']} {message}", style="info")


def format_discovery(message: str) -> Text:
    """Format a discovery message for real-time insights."""
    return Text(f"{SYMBOLS['discovery_text']} {message}", style="primary")


def format_analysis(message: str) -> Text:
    """Format an analysis message for decomposition results."""
    return Text(f"{SYMBOLS['analysis_text']} {message}", style="info")


def format_recommendation(message: str) -> Text:
    """Format a recommendation message for actionable insights."""
    return Text(f"{SYMBOLS['recommendation_text']} {message}", style="success")