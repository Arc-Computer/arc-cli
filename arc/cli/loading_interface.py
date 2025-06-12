"""Arc Loading Interface - Visual Feedback for Config Analysis.

Provides professional loading states and progress tracking for agent
configuration analysis and capability extraction.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc.cli.design_standards import PROGRESS_STYLES, COLORS, LAYOUT
from arc.cli.message_templates import COMMAND_MESSAGES, format_currency


class ConfigAnalysisLoader:
    """Loading interface for agent configuration analysis."""
    
    def __init__(self, console: Console):
        """Initialize with console for output."""
        self.console = console
        self.current_progress = None
        
    async def analyze_config_with_progress(self, config_path: str, 
                                         parser, normalizer) -> Dict[str, Any]:
        """
        Analyze agent configuration with visual progress feedback.
        
        Args:
            config_path: Path to configuration file
            parser: AgentConfigParser instance
            normalizer: ConfigNormalizer instance
            
        Returns:
            Complete agent profile with capabilities
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        ) as progress:
            
            # Phase 1: Parse Configuration
            parse_task = progress.add_task("Loading agent configuration...", total=None)
            await asyncio.sleep(0.5)  # Simulate parsing time
            
            try:
                config = parser.parse(config_path)
                progress.update(parse_task, description="[success]✓[/success] Configuration loaded")
                await asyncio.sleep(0.2)
            except Exception as e:
                progress.update(parse_task, description=f"[error]✗[/error] Parse failed: {str(e)}")
                raise
            
            # Phase 2: Extract Capabilities
            capabilities_task = progress.add_task("Analyzing agent capabilities...", total=None)
            await asyncio.sleep(0.8)  # Simulate capability analysis
            
            try:
                capabilities = parser.extract_capabilities(config)
                progress.update(capabilities_task, description="[success]✓[/success] Capabilities extracted")
                await asyncio.sleep(0.2)
            except Exception as e:
                progress.update(capabilities_task, description=f"[error]✗[/error] Analysis failed: {str(e)}")
                raise
            
            # Phase 3: Normalize Configuration
            normalize_task = progress.add_task("Normalizing for Arc processing...", total=None)
            await asyncio.sleep(0.4)  # Simulate normalization
            
            try:
                normalized_config = normalizer.normalize(config)
                progress.update(normalize_task, description="[success]✓[/success] Configuration normalized")
                await asyncio.sleep(0.2)
            except Exception as e:
                progress.update(normalize_task, description=f"[error]✗[/error] Normalization failed: {str(e)}")
                raise
            
            # Phase 4: Create Agent Profile
            profile_task = progress.add_task("Building agent profile...", total=None)
            await asyncio.sleep(0.3)  # Simulate profile creation
            
            try:
                # Create comprehensive agent profile
                agent_profile = {
                    "configuration": normalized_config,
                    "capabilities": capabilities,
                    "normalizer_enhancements": getattr(normalizer, 'enhancements', []),
                    "config_path": config_path,
                    "analysis_timestamp": time.time()
                }
                progress.update(profile_task, description="[success]✓[/success] Agent profile ready")
                await asyncio.sleep(0.1)
            except Exception as e:
                progress.update(profile_task, description=f"[error]✗[/error] Profile creation failed: {str(e)}")
                raise
        
        return agent_profile
    
    def display_config_summary(self, agent_profile: Dict[str, Any]) -> None:
        """Display configuration analysis summary."""
        config = agent_profile.get("configuration", {})
        capabilities = agent_profile.get("capabilities", {})
        
        # Create summary table
        summary_table = Table(title="Agent Configuration Summary", show_header=True)
        summary_table.add_column("Property", style="info", width=20)
        summary_table.add_column("Value", style="#3B82F6")
        
        # Basic configuration
        summary_table.add_row("Model", config.get("model", "Unknown"))
        summary_table.add_row("Temperature", str(config.get("temperature", "Unknown")))
        summary_table.add_row("Tool Count", str(len(config.get("tools", []))))
        
        # Capabilities
        domains = capabilities.get("domains", ["general"])
        summary_table.add_row("Domains", ", ".join(domains))
        summary_table.add_row("Complexity", capabilities.get("complexity_level", "Unknown"))
        
        # Tool categories
        tool_categories = capabilities.get("tool_categories", {})
        if tool_categories:
            categories = ", ".join(tool_categories.keys())
            summary_table.add_row("Tool Categories", categories)
        
        self.console.print()
        self.console.print(summary_table)
        self.console.print()
        
        # Display any warnings
        enhancements = agent_profile.get("normalizer_enhancements", [])
        if enhancements:
            self.console.print("[warning]Configuration Enhancements Applied:[/warning]")
            for enhancement in enhancements[:5]:  # Show first 5
                self.console.print(f"  [muted]•[/muted] {enhancement}")
            if len(enhancements) > 5:
                self.console.print(f"  [muted]... and {len(enhancements) - 5} more[/muted]")
            self.console.print()


class ExecutionProgressLoader:
    """Loading interface for real-time execution progress."""
    
    def __init__(self, console: Console):
        """Initialize with console for output."""
        self.console = console
        self.live_progress = None
        
    def create_execution_progress(self, scenario_count: int) -> Progress:
        """Create progress tracker for scenario execution."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=LAYOUT["progress_bar_width"]),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            expand=False
        )
        
        return progress
    
    def display_live_metrics(self, metrics: Dict[str, Any]) -> Panel:
        """Display live execution metrics in enhanced tree format."""
        content_lines = []
        
        # Header with rocket emoji
        total = metrics.get("total", 0)
        header = f"🚀 Executing {total} scenarios via Modal..."
        content_lines.append(header)
        
        # Container visualization (if available)
        if "active_containers" in metrics:
            active = metrics["active_containers"]
            max_containers = metrics.get("max_containers", active)
            
            # Container bar visualization
            bar_width = 20
            if max_containers > 0:
                filled = int(bar_width * active / max_containers)
                bar = "█" * filled + "░" * (bar_width - filled)
            else:
                bar = "░" * bar_width
            
            content_lines.append(f"├── Containers: {bar} {active}/{max_containers} active")
        
        # Progress metrics
        if "completed" in metrics and "total" in metrics:
            completed = metrics["completed"]
            total = metrics["total"]
            percentage = (completed / total * 100) if total > 0 else 0
            
            # Progress bar visualization
            bar_width = 20
            filled = int(bar_width * percentage / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            content_lines.append(f"├── Progress:   {bar} {completed}/{total} complete ({percentage:.0f}%)")
        
        # Cost tracking
        if "cost" in metrics:
            cost = metrics["cost"]
            estimated = metrics.get("estimated_cost", cost * 1.2)
            content_lines.append(f"├── Cost:       {format_currency(cost)} (estimated: {format_currency(estimated)})")
        
        # Time tracking
        if "elapsed_time" in metrics:
            elapsed = metrics["elapsed_time"]
            estimated_total = metrics.get("estimated_total_time", elapsed * 1.5)
            content_lines.append(f"├── Time:       {self._format_duration(elapsed)} (estimated: {self._format_duration(estimated_total)})")
        
        # Failures detected (use └── for last item)
        if "failures" in metrics:
            failures = metrics["failures"]
            status = "clustering in progress..." if failures > 0 else "none detected"
            content_lines.append(f"└── Failures:   {failures} detected, {status}")
        elif content_lines and content_lines[-1].startswith("├──"):
            # Replace last ├── with └── if failures not present
            content_lines[-1] = content_lines[-1].replace("├──", "└──")
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="[#3B82F6]Real-time Execution Monitoring[/#3B82F6]",
            border_style="#3B82F6",
            padding=(0, 1)
        )
    
    def display_reliability_breakdown(self, reliability_data: Dict[str, Any]) -> Panel:
        """Display real-time reliability score breakdown."""
        overall_score = reliability_data.get("overall_score", 0)
        dimension_scores = reliability_data.get("dimension_scores", {})
        
        content_lines = [
            f"Reliability Score: {overall_score:.0f}% ({reliability_data.get('completed', 0)}/{reliability_data.get('total', 0)} scenarios)"
        ]
        
        if dimension_scores:
            content_lines.append("─" * 45)
            
            for dimension, score in dimension_scores.items():
                # Create visual bar for each dimension
                bar_width = 10
                filled = int(bar_width * score / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                # Weight information
                weight = reliability_data.get("dimension_weights", {}).get(dimension, 20)
                
                content_lines.append(f"{dimension:<18} {bar} {score}/10 ({weight}%)")
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="[success]Reliability Analysis[/success]",
            border_style="success",
            padding=(0, 1)
        )
    
    def display_assumption_alerts(self, violations: list) -> Optional[Panel]:
        """Display real-time assumption violation alerts."""
        if not violations:
            return None
        
        content_lines = []
        
        for violation in violations[:3]:  # Show top 3
            severity_color = {
                "critical": "error",
                "high": "warning", 
                "medium": "info",
                "low": "muted"
            }.get(violation.get("severity", "low"), "muted")
            
            violation_type = violation.get("type", "unknown").upper()
            description = violation.get("description", "")[:60] + "..."
            
            content_lines.append(f"[{severity_color}]{violation_type}[/{severity_color}]: {description}")
        
        if len(violations) > 3:
            content_lines.append(f"[muted]... and {len(violations) - 3} more violations[/muted]")
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="[warning]Assumption Violations Detected[/warning]",
            border_style="warning",
            padding=(0, 1)
        )

    def display_live_error_monitoring(self, error_data: Dict[str, Any]) -> Optional[Panel]:
        """Display live error monitoring and recovery procedures."""
        if not error_data.get("errors"):
            return None
            
        content_lines = []
        
        # Error monitoring header
        content_lines.append("🚨 Live Error Monitoring & Recovery")
        content_lines.append("")
        
        # Error summary
        total_errors = error_data.get("total_errors", 0)
        error_rate = error_data.get("error_rate", 0)
        content_lines.append(f"├── Total Errors: {total_errors}")
        content_lines.append(f"├── Error Rate: {error_rate:.1f}%")
        
        # Top error categories
        error_categories = error_data.get("error_categories", {})
        if error_categories:
            content_lines.append("├── Top Error Types:")
            for category, count in list(error_categories.items())[:3]:
                content_lines.append(f"│   ├── {category}: {count} occurrences")
        
        # Recovery procedures
        recovery_status = error_data.get("recovery_status", "none")
        recovery_icon = "🔄" if recovery_status == "in_progress" else "✅" if recovery_status == "completed" else "⏸️"
        content_lines.append(f"├── Recovery Status: {recovery_icon} {recovery_status}")
        
        # Error clustering
        clustering_active = error_data.get("clustering_active", False)
        if clustering_active:
            clusters_found = error_data.get("clusters_found", 0)
            content_lines.append(f"└── Error Clustering: 🔍 {clusters_found} patterns identified")
        else:
            content_lines.append("└── Error Clustering: ⏹️ inactive")
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="[error]Error Monitoring & Recovery[/error]",
            border_style="error",
            padding=(0, 1)
        )
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def display_execution_timeline(self, timeline_data: Dict[str, Any]) -> Panel:
        """Display execution timeline with performance metrics."""
        content_lines = []
        
        # Timeline header
        content_lines.append("📊 Execution Timeline & Performance")
        content_lines.append("")
        
        # Key performance metrics
        throughput = timeline_data.get("scenarios_per_minute", 0)
        avg_latency = timeline_data.get("avg_scenario_time", 0)
        speedup_factor = timeline_data.get("speedup_factor", 1)
        
        content_lines.append(f"├── Throughput:     {throughput:.1f} scenarios/min")
        content_lines.append(f"├── Avg Latency:    {avg_latency:.2f}s per scenario")
        content_lines.append(f"├── Speedup Factor: {speedup_factor:.1f}x vs sequential")
        
        # Container utilization over time
        if "container_efficiency" in timeline_data:
            efficiency = timeline_data["container_efficiency"]
            content_lines.append(f"├── Container Efficiency: {efficiency:.1f}%")
        
        # Recent performance trend
        trend = timeline_data.get("performance_trend", "stable")
        trend_icon = "📈" if trend == "improving" else "📉" if trend == "degrading" else "➡️"
        content_lines.append(f"└── Performance Trend: {trend_icon} {trend}")
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="[#10B981]Performance Analytics[/#10B981]",
            border_style="#10B981",
            padding=(0, 1)
        )


class StreamingResultsDisplay:
    """Display streaming results as they arrive."""
    
    def __init__(self, console: Console):
        """Initialize with console for output."""
        self.console = console
        self.results_buffer = []
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a new result to the streaming display."""
        self.results_buffer.append(result)
        
        # Keep only last 10 results to prevent overwhelming display
        if len(self.results_buffer) > 10:
            self.results_buffer.pop(0)
    
    def create_results_table(self) -> Table:
        """Create table showing recent results."""
        table = Table(title="Recent Results", show_header=True)
        table.add_column("Scenario", style="info", width=15)
        table.add_column("Status", width=8)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Issue", style="muted", width=30)
        
        for result in self.results_buffer:
            scenario_id = result.get("scenario_id", "unknown")[:12]
            
            # Status with color
            success = result.get("success", False)
            status = "[success]PASS[/success]" if success else "[error]FAIL[/error]"
            
            # Score
            score = result.get("reliability_score", 0)
            score_text = f"{score:.0f}%"
            
            # Primary issue (if any)
            issue = result.get("primary_issue", "None")[:25] if not success else ""
            
            table.add_row(scenario_id, status, score_text, issue)
        
        return table