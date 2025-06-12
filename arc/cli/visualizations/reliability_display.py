"""Enhanced CLI visualization for 5-dimensional reliability scoring with explanations."""

from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import BarColumn, Progress, TaskID
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align
from rich.padding import Padding

from arc.sandbox.evaluation.reliability_scorer import ReliabilityScore


class ReliabilityDisplayManager:
    """Manages enhanced CLI visualization for 5-dimensional reliability scoring."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        
        # Colors for different score ranges
        self.score_colors = {
            "excellent": "bright_green",
            "good": "green", 
            "average": "yellow",
            "poor": "red",
            "critical": "bright_red"
        }
        
        # Dimension display settings
        self.dimensions = {
            "tool_execution": {"name": "Tool Execution", "weight": 30, "icon": "🔧"},
            "response_quality": {"name": "Response Quality", "weight": 25, "icon": "📝"}, 
            "error_handling": {"name": "Error Handling", "weight": 20, "icon": "🛡️"},
            "performance": {"name": "Performance", "weight": 15, "icon": "⚡"},
            "completeness": {"name": "Completeness", "weight": 10, "icon": "✅"}
        }
    
    def display_reliability_dashboard(
        self, 
        reliability_score: ReliabilityScore,
        scenario_count: int,
        success_count: int,
        root_cause_analysis: Optional[Dict[str, Any]] = None,
        minimal_repro: Optional[str] = None
    ) -> None:
        """Display the comprehensive reliability dashboard."""
        
        # Main scoring panel
        self._display_main_scoring_panel(reliability_score, scenario_count, success_count)
        
        # Dimension breakdown
        self._display_dimension_breakdown(reliability_score)
        
        # Root cause analysis if available
        if root_cause_analysis:
            self._display_root_cause_analysis(root_cause_analysis, minimal_repro)
        
        # Recommendations
        if reliability_score.recommendations:
            self._display_recommendations(reliability_score.recommendations)
    
    def _display_main_scoring_panel(
        self, 
        reliability_score: ReliabilityScore,
        scenario_count: int, 
        success_count: int
    ) -> None:
        """Display the main reliability score panel matching the issue format."""
        
        overall_percentage = int(reliability_score.overall_score)
        grade = reliability_score._get_grade()
        
        # Create the main panel content matching the format from the issue
        header = f"Reliability Score: {overall_percentage}% ({success_count}/{scenario_count} scenarios)"
        
        # Create the visual panel
        panel_content = []
        panel_content.append(f"[bold bright_cyan]{header}[/bold bright_cyan]")
        panel_content.append("─" * len(header))
        
        # Add dimension bars
        for dim_key, dim_info in self.dimensions.items():
            score = reliability_score.dimension_scores.get(dim_key, 0)
            weight = dim_info["weight"]
            name = dim_info["name"]
            
            # Create progress bar visualization
            bar_filled = int(score / 10)  # 0-10 scale
            bar_empty = 10 - bar_filled
            bar_visual = "█" * bar_filled + "░" * bar_empty
            
            score_color = self._get_score_color(score)
            
            panel_content.append(
                f"{name:<18} [{score_color}]{bar_visual}[/{score_color}] "
                f"{int(score)}/10 ({weight}%)"
            )
        
        # Create bordered panel
        panel = Panel(
            "\n".join(panel_content),
            border_style="bright_cyan",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()
    
    def _display_dimension_breakdown(self, reliability_score: ReliabilityScore) -> None:
        """Display detailed breakdown of each dimension with explanations."""
        
        self.console.print("[bold bright_cyan]📊 Dimension Analysis[/bold bright_cyan]")
        self.console.print()
        
        # Create table for detailed breakdown
        table = Table(show_header=True, header_style="bold bright_cyan")
        table.add_column("Dimension", style="bright_white", width=20)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Grade", justify="center", width=6)
        table.add_column("Weight", justify="center", width=8)
        table.add_column("Issues Found", width=40)
        
        for dim_key, dim_info in self.dimensions.items():
            score = reliability_score.dimension_scores.get(dim_key, 0)
            grade = self._score_to_grade(score)
            weight = f"{dim_info['weight']}%"
            icon = dim_info["icon"]
            name = dim_info["name"]
            
            # Find issues for this dimension
            dimension_issues = [
                issue for issue in reliability_score.issues_found
                if issue.get("dimension") == dim_key
            ]
            
            issues_text = ""
            if dimension_issues:
                issues_text = f"{len(dimension_issues)} issues: "
                issues_text += ", ".join([
                    issue.get("issue", "Unknown")[:30] + "..."
                    if len(issue.get("issue", "")) > 30
                    else issue.get("issue", "Unknown")
                    for issue in dimension_issues[:2]  # Show first 2 issues
                ])
            else:
                issues_text = "No issues detected"
            
            score_color = self._get_score_color(score)
            grade_color = self._get_grade_color(grade)
            
            table.add_row(
                f"{icon} {name}",
                f"[{score_color}]{score:.0f}[/{score_color}]",
                f"[{grade_color}]{grade}[/{grade_color}]",
                weight,
                issues_text
            )
        
        self.console.print(table)
        self.console.print()
    
    def _display_root_cause_analysis(
        self, 
        root_cause: Dict[str, Any],
        minimal_repro: Optional[str] = None
    ) -> None:
        """Display root cause analysis section."""
        
        self.console.print("[bold red]🔍 Root Cause Analysis[/bold red]")
        self.console.print()
        
        # Primary failure info
        primary_failure = root_cause.get("primary_failure", "Unknown")
        capability = root_cause.get("capability", "Unknown") 
        assumption = root_cause.get("assumption", "Unknown")
        impact = root_cause.get("impact", "Unknown")
        fix_suggestion = root_cause.get("fix_suggestion", "No suggestion available")
        
        # Create structured display
        cause_lines = []
        cause_lines.append(f"[bold red]Primary Failure:[/bold red] {primary_failure}")
        cause_lines.append(f"├── [yellow]Capability:[/yellow] {capability}")
        cause_lines.append(f"├── [yellow]Assumption:[/yellow] \"{assumption}\"")
        cause_lines.append(f"├── [red]Impact:[/red] {impact}")
        cause_lines.append(f"└── [green]Fix:[/green] {fix_suggestion}")
        
        panel = Panel(
            "\n".join(cause_lines),
            border_style="red",
            title="Root Cause Attribution",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Minimal reproduction if available
        if minimal_repro:
            self.console.print()
            self.console.print("[bold yellow]📝 Minimal Reproduction[/bold yellow]")
            
            repro_panel = Panel(
                minimal_repro,
                border_style="yellow",
                title="Minimal Repro",
                padding=(1, 2)
            )
            self.console.print(repro_panel)
        
        self.console.print()
    
    def _display_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Display actionable recommendations."""
        
        self.console.print("[bold green]💡 Recommendations[/bold green]")
        self.console.print()
        
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            priority = rec.get("priority", "medium")
            dimension = rec.get("dimension", "general")
            recommendation = rec.get("recommendation", "No recommendation")
            actions = rec.get("specific_actions", [])
            
            priority_color = {
                "critical": "bright_red",
                "high": "red", 
                "medium": "yellow",
                "low": "green"
            }.get(priority, "white")
            
            # Format recommendation
            rec_text = []
            rec_text.append(f"[{priority_color}]{priority.upper()} PRIORITY[/{priority_color}]")
            rec_text.append(f"[bold]Dimension:[/bold] {dimension.replace('_', ' ').title()}")
            rec_text.append(f"[bold]Recommendation:[/bold] {recommendation}")
            
            if actions:
                rec_text.append("[bold]Specific Actions:[/bold]")
                for action in actions:
                    rec_text.append(f"  • {action}")
            
            panel = Panel(
                "\n".join(rec_text),
                border_style=priority_color,
                title=f"Recommendation {i}",
                padding=(1, 2)
            )
            
            self.console.print(panel)
        
        self.console.print()
    
    def _get_score_color(self, score: float) -> str:
        """Get color for score based on value."""
        if score >= 90:
            return self.score_colors["excellent"]
        elif score >= 80:
            return self.score_colors["good"]
        elif score >= 70:
            return self.score_colors["average"]
        elif score >= 60:
            return self.score_colors["poor"]
        else:
            return self.score_colors["critical"]
    
    def _get_grade_color(self, grade: str) -> str:
        """Get color for grade."""
        grade_colors = {
            "A": "bright_green",
            "B": "green",
            "C": "yellow", 
            "D": "red",
            "F": "bright_red"
        }
        return grade_colors.get(grade, "white")
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


def create_reliability_dashboard_from_results(
    results: List[Dict[str, Any]],
    scenario_count: int,
    console: Optional[Console] = None
) -> None:
    """Create and display reliability dashboard from execution results."""
    
    # Calculate aggregate reliability data
    success_count = sum(1 for r in results if r.get("success", False))
    
    # Extract reliability scores from results
    reliability_scores = []
    dimension_totals = {}
    all_issues = []
    all_recommendations = []
    
    for result in results:
        if "reliability_score" in result:
            rel_score = result["reliability_score"]
            if isinstance(rel_score, dict):
                reliability_scores.append(rel_score)
                
                # Aggregate dimension scores
                for dim, score in rel_score.get("dimension_scores", {}).items():
                    if dim not in dimension_totals:
                        dimension_totals[dim] = []
                    dimension_totals[dim].append(score)
                
                # Collect issues and recommendations
                all_issues.extend(rel_score.get("issues_found", []))
                all_recommendations.extend(rel_score.get("recommendations", []))
    
    # Calculate average scores
    avg_dimension_scores = {}
    for dim, scores in dimension_totals.items():
        avg_dimension_scores[dim] = sum(scores) / len(scores) if scores else 0
    
    # Calculate overall score
    weights = {
        "tool_execution": 0.3,
        "response_quality": 0.25,
        "error_handling": 0.2,
        "performance": 0.15,
        "completeness": 0.1
    }
    
    overall_score = sum(
        avg_dimension_scores.get(dim, 0) * weight
        for dim, weight in weights.items()
    )
    
    # Create aggregate ReliabilityScore object
    from arc.sandbox.evaluation.reliability_scorer import ReliabilityScore
    
    aggregate_score = ReliabilityScore(
        overall_score=overall_score,
        dimension_scores=avg_dimension_scores,
        issues_found=all_issues,
        recommendations=all_recommendations
    )
    
    # Generate root cause analysis using failure attributor
    from arc.sandbox.evaluation.failure_attributor import analyze_failure_patterns
    
    # Extract failed results for root cause analysis
    failed_results = [r for r in results if not r.get("success", False)]
    
    if failed_results:
        failure_analysis = analyze_failure_patterns(failed_results)
        
        primary_issue = failure_analysis.get("primary_issue", {})
        root_cause = {
            "primary_failure": primary_issue.get("capability", "Unknown failure type").replace("_", " ").title(),
            "capability": primary_issue.get("capability", "Unknown capability"), 
            "assumption": primary_issue.get("assumption", "Unknown assumption"),
            "impact": f"{failure_analysis['total_failures']} failures affecting {scenario_count - success_count} scenarios",
            "fix_suggestion": f"Address {primary_issue.get('capability', 'capability')} issues. See recommendations below."
        }
    else:
        # No failures - no root cause analysis needed
        root_cause = None
    
    # Generate minimal repro from first failure if available
    minimal_repro = None
    if failed_results and failure_analysis.get("attributions"):
        first_attribution = failure_analysis["attributions"][0]
        minimal_repro = first_attribution.minimal_repro
    
    # Display dashboard
    display_manager = ReliabilityDisplayManager(console)
    display_manager.display_reliability_dashboard(
        reliability_score=aggregate_score,
        scenario_count=scenario_count,
        success_count=success_count,
        root_cause_analysis=root_cause,
        minimal_repro=minimal_repro
    )