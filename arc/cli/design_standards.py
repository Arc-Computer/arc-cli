"""Arc Enterprise CLI Design Standards.

Professional visual standards for enterprise-grade CLI interface with
streaming-ready components and assumption validation focus.
"""

from rich.progress import (
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TransferSpeedColumn
)

# Enterprise color palette - professional and accessible
COLORS = {
    'primary': '#3B82F6',          # Headers, Arc branding, key insights (bright blue)
    'success': '#10B981',          # Improvements, positive metrics, PASS indicators (green)
    'warning': '#F59E0B',          # Attention items, assumption violations (yellow)
    'error': '#EF4444',            # Failures, critical issues, FAIL indicators (red)
    'info': '#06B6D4',             # Statistical data, metadata, secondary info (cyan)
    'muted': '#6B7280',            # Secondary text, supporting details (gray)
    'accent': '#8B5CF6',           # Highlights, key values, important numbers (purple)
}

# Layout standards optimized for streaming and assumption highlighting
LAYOUT = {
    'terminal_width': 120,
    'progress_bar_width': 40,
    'panel_padding': (0, 1),
    'live_update_area': (0, 2),    # Space reserved for streaming content
    'static_header': (0, 1),       # Fixed header area during streaming
    'progress_section': (2, 1),    # Dedicated progress display area
    'assumption_highlight': 'bold bright_yellow',  # Critical for assumption violations
    'funnel_step_width': 25,       # Consistent funnel visualization width
    'statistical_precision': 2,    # Decimal places for confidence intervals
}

# Progress indicators designed for real-time feedback and streaming
PROGRESS_STYLES = {
    'scenario_generation': [SpinnerColumn(), TextColumn("[progress.description]{task.description}")],
    'execution': [BarColumn(), TimeRemainingColumn(), TransferSpeedColumn()],
    'analysis': [SpinnerColumn(), TextColumn("[progress.description]{task.description}")],
    'streaming_results': [BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()],
    'assumption_detection': [SpinnerColumn(), TextColumn("Scanning for assumption violations...")],
    'model_optimization': [SpinnerColumn(), TextColumn("Analyzing model performance...")],
    'funnel_building': [BarColumn(), TextColumn("Building capability funnel...")],
}

# Professional symbols - only allowed emojis per CLAUDE.md requirements
SYMBOLS = {
    'pass': '✓',          # Success indicator (allowed)
    'fail': '✗',          # Failure indicator (allowed)  
    'bullet': '•',        # List bullets (allowed)
    'up': '↑',           # Improvements, increases (allowed)
    'down': '↓',         # Decreases, degradation (allowed)
    'right': '→',        # Flow, progression (allowed)
    
    # Professional text replacements for forbidden emojis
    'pass_text': 'PASS',
    'fail_text': 'FAIL',
    'warning_text': 'ATTENTION',
    'info_text': 'INFO',
    'success_text': 'SUCCESS',
    'error_text': 'ERROR',
    'discovery_text': 'DISCOVERED',
    'analysis_text': 'ANALYSIS',
    'recommendation_text': 'RECOMMENDED',
}

# Assumption validation styles - core to Arc's value proposition
ASSUMPTION_STYLES = {
    'violation_header': 'bold bright_yellow on black',
    'violation_detail': 'bright_yellow',
    'violation_impact': 'bold bright_red',
    'violation_fix': 'bright_green',
    'assumption_pattern': 'italic bright_cyan',
}

# Funnel visualization standards for capability decomposition
FUNNEL_STYLES = {
    'step_header': 'bold bright_blue',
    'step_success': 'bright_green',
    'step_failure': 'bright_red',
    'step_rate': 'bright_cyan',
    'bottleneck': 'bold bright_yellow',
    'flow_arrow': 'bright_black',
}

# Statistical display standards - enterprise credibility requirement
STATISTICAL_STYLES = {
    'confidence_high': 'bold bright_green',    # p < 0.01, high confidence
    'confidence_medium': 'bright_yellow',      # p < 0.05, medium confidence  
    'confidence_low': 'bright_red',            # p >= 0.05, low confidence
    'p_value': 'italic bright_cyan',
    'effect_size': 'bold bright_magenta',
    'sample_size': 'bright_black',
}

# Business impact translation styles
BUSINESS_STYLES = {
    'cost_savings': 'bold bright_green',
    'time_savings': 'bright_green', 
    'risk_reduction': 'bright_cyan',
    'incident_prevention': 'bold bright_blue',
    'roi_positive': 'bold bright_green',
    'roi_negative': 'bright_red',
}

# Real-time streaming display configuration
STREAMING_CONFIG = {
    'refresh_rate': 0.1,           # 10fps for smooth updates
    'buffer_lines': 5,             # Lines to reserve for streaming content
    'auto_scroll': True,           # Scroll to show latest results
    'preserve_header': True,       # Keep header visible during streaming
    'cost_update_interval': 1.0,   # Update cost display every second
    'progress_smoothing': True,    # Smooth progress bar transitions
}

# Enterprise messaging configuration
MESSAGING_CONFIG = {
    'max_line_length': 100,        # Consistent line wrapping
    'indent_size': 2,              # Standard indentation
    'section_spacing': 1,          # Lines between sections
    'emphasis_style': 'bold',      # Standard emphasis
    'code_theme': 'monokai',       # Syntax highlighting theme
}