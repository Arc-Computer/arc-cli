# Arc Enterprise CLI Implementation Plan

## Overview
Implement enterprise-grade CLI interface that transforms technical output into actionable business insights, focusing on assumption validation and rapid iteration cycles.

## Phase 1: Foundation (1 hour)

### 1. Design Standards Module (`arc/cli/design_standards.py`)
```python
# Core visual identity and layout standards
COLORS = {
    'primary': 'bright_blue',      # Headers, Arc branding
    'success': 'bright_green',     # Improvements, positive metrics  
    'warning': 'bright_yellow',    # Attention items, assumptions
    'error': 'bright_red',         # Failures, critical issues
    'info': 'bright_cyan',         # Statistical data
    'muted': 'bright_black',       # Secondary text
    'accent': 'bright_magenta',    # Highlights, key values
}

LAYOUT = {
    'terminal_width': 120,
    'progress_bar_width': 40,
    'panel_padding': (0, 1),
    'assumption_highlight': 'bold bright_yellow',  # For surfacing hidden assumptions
}

# Progress indicators for real-time feedback
PROGRESS_STYLES = {
    'scenario_generation': SpinnerColumn() + TextColumn(),
    'execution': BarColumn() + TimeRemainingColumn(),
    'analysis': SpinnerColumn() + TextColumn("[progress.description]{task.description}"),
}
```

### 2. Enterprise Messaging (`arc/cli/enterprise_messaging.py`)
```python
# Assumption-focused messaging
ASSUMPTION_MESSAGES = {
    'currency': "ASSUMPTION VIOLATED: Agent assumes {default_currency} for all transactions",
    'language': "ASSUMPTION VIOLATED: Agent only handles {default_language} input",
    'timezone': "ASSUMPTION VIOLATED: Agent uses {default_timezone} without validation",
}

# Action-oriented messages
ACTION_MESSAGES = {
    'infrastructure_fix': "INFRASTRUCTURE FIX: {description} resolves {percentage}% of failures",
    'model_switch': "MODEL OPTIMIZATION: Switch to {model} for {cost_reduction}% cost savings",
    'config_change': "CONFIGURATION FIX: {change} improves reliability by {improvement} pp",
}

# Business impact translations
IMPACT_MESSAGES = {
    'incident_prevention': "${amount:,} in prevented incidents per month",
    'time_savings': "{hours} engineering hours saved vs reactive debugging",
    'trust_impact': "{percentage}% reduction in customer-facing errors",
}
```

### 3. Console Enhancement (`arc/cli/utils/console.py`)
Update existing ArcConsole class:
- Remove emoji usage (✓ → "PASS", ✗ → "FAIL")
- Add assumption highlighting methods
- Add funnel visualization helpers
- Add statistical formatting (p-values, CI)

## Phase 2: Multi-Model Intelligence (1 hour)

### 4. Model Recommendation Engine (`arc/recommendations/model_optimizer.py`)
```python
class ModelOptimizer:
    """Infrastructure-first approach to model selection"""
    
    def analyze_for_assumptions(self, failures: List[Failure]) -> AssumptionReport:
        """Identify if failures are due to model limitations or config assumptions"""
        # Decompose failures by type
        # Separate model issues from infrastructure issues
        # Prioritize config fixes over model changes
    
    def recommend_model_if_needed(self, current_perf: Performance) -> Optional[ModelRec]:
        """Only recommend model change if config fixes won't solve the problem"""
        # Check if infrastructure fixes available first
        # Calculate cost/performance tradeoffs
        # Include confidence based on test coverage
```

### 5. Failure Decomposition (`arc/analysis/funnel_analyzer.py`)
```python
class FunnelAnalyzer:
    """Decompose agent execution into measurable steps"""
    
    def build_capability_funnel(self, trajectories: List[Trajectory]) -> Funnel:
        """Create step-by-step success funnel"""
        # Extract execution steps
        # Calculate success rate at each step
        # Identify bottleneck steps
        
    def find_assumption_violations(self, funnel: Funnel) -> List[Assumption]:
        """Identify hidden assumptions causing failures"""
        # Pattern match for common assumptions
        # Extract minimal reproductions
        # Group by assumption type
```

## Phase 3: Command Updates (2 hours)

### 6. Enhanced Command Outputs

#### `arc run` - Proactive Discovery Focus
- Real-time assumption violation detection
- Running count of "issues found BEFORE production"
- Capability funnel preview
- Infrastructure vs model issue breakdown

#### `arc analyze` - Origin of Failure Focus  
- Capability funnel visualization
- Assumption violation clustering
- Minimal reproduction with complexity reduction
- "Why" before "what" in failure reporting

#### `arc recommend` - Action Priority Focus
- Infrastructure fixes first (config, prompts, tools)
- Model changes only when necessary
- Implementation time estimates
- Confidence scores based on similar fixes

#### `arc diff` - Assumption Validation Focus
- Before/after assumption handling
- Statistical validation of improvements
- Funnel comparison (where improvements happened)
- Business impact quantification

#### `arc status` - Measurement Dashboard
- Assumption violation trends
- Infrastructure vs model fix ratio
- Time-to-fix metrics
- Cumulative value delivered

### 7. Error Handling (`arc/cli/error_handlers.py`)
- Actionable error messages
- Suggest infrastructure fixes first
- Link errors to assumptions
- Provide fallback options

## Implementation Order

1. **Hour 1**: Foundation
   - Design standards
   - Enterprise messaging
   - Console utilities update

2. **Hour 2**: Intelligence Layer
   - Model optimizer (infrastructure-first)
   - Funnel analyzer
   - Assumption detector

3. **Hour 3**: Command Updates
   - Update run command
   - Update analyze command
   - Update recommend command

4. **Hour 4**: Polish & Testing
   - Update diff and status commands
   - Error handling
   - Integration testing
   - Demo preparation

## Key Integration Points

### With Database (Issue #37)
- Store assumption violations for pattern learning
- Track which fixes actually worked
- Build historical funnel data

### With Modal (Issue #5)
- Real-time funnel updates during execution
- Stream assumption violations as discovered
- Cost tracking with ROI calculation

### With Scenario Generation (Issue #7)
- Generate scenarios targeting discovered assumptions
- Test assumption fixes specifically
- Validate infrastructure improvements

## Success Criteria

1. **Assumption Visibility**: Hidden assumptions surface within first run
2. **Action Clarity**: Every output includes specific next step
3. **Infrastructure First**: 80% of fixes are config/prompt changes
4. **Rapid Iteration**: < 5 min from problem to validated fix
5. **Business Translation**: Every technical metric has dollar impact

## Demo Script Integration

```bash
# Shows assumption discovery
arc run finance_agent_v1.yaml
# "PROACTIVE DISCOVERY: 12 currency assumption violations found BEFORE production"

# Shows failure decomposition  
arc analyze
# "FUNNEL ANALYSIS: 86% of failures occur at currency parsing step"

# Shows infrastructure-first fix
arc recommend  
# "INFRASTRUCTURE FIX: Add currency validation to prompt (10 min implementation)"

# Shows validated improvement
arc diff finance_agent_v1.yaml finance_agent_v2.yaml
# "ASSUMPTION FIXED: Currency handling improved by 91.7%"
```

## Files to Create/Modify

**New Files:**
- `arc/cli/design_standards.py`
- `arc/cli/enterprise_messaging.py`
- `arc/recommendations/model_optimizer.py`
- `arc/analysis/funnel_analyzer.py`
- `arc/cli/error_handlers.py`

**Modified Files:**
- `arc/cli/utils/console.py` (remove emojis, add methods)
- `arc/cli/commands/run.py` (new output format)
- `arc/cli/commands/analyze.py` (funnel visualization)
- `arc/cli/commands/recommend.py` (infrastructure-first)
- `arc/cli/commands/diff.py` (assumption validation)
- `arc/cli/commands/status.py` (measurement dashboard)

## Testing Focus

1. **Assumption Detection**: Verify currency assumptions surface clearly
2. **Funnel Accuracy**: Steps decompose meaningfully
3. **Action Validation**: Recommendations are implementable
4. **Statistical Rigor**: All claims have valid backing
5. **Performance**: Real-time updates work smoothly