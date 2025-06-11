"""Arc CLI Message Templates.

Professional messaging that translates technical findings into
actionable insights with statistical credibility and clear communication.
"""

# Assumption violation messages - core to Arc's value proposition
ASSUMPTION_MESSAGES = {
    'currency': "ASSUMPTION VIOLATED: Agent assumes {default_currency} for all transactions",
    'language': "ASSUMPTION VIOLATED: Agent only handles {default_language} input", 
    'timezone': "ASSUMPTION VIOLATED: Agent uses {default_timezone} without validation",
    'encoding': "ASSUMPTION VIOLATED: Agent expects {default_encoding} text encoding",
    'format': "ASSUMPTION VIOLATED: Agent assumes {default_format} data format",
    'precision': "ASSUMPTION VIOLATED: Agent uses {default_precision} decimal precision",
    'scale': "ASSUMPTION VIOLATED: Agent assumes {default_scale} data ranges",
    'authentication': "ASSUMPTION VIOLATED: Agent bypasses {auth_type} authentication",
    'validation': "ASSUMPTION VIOLATED: Agent skips input validation for {input_type}",
    'error_handling': "ASSUMPTION VIOLATED: Agent fails gracefully without proper error handling",
}

# Action-oriented messages prioritizing infrastructure over intelligence
ACTION_MESSAGES = {
    'infrastructure_fix': "INFRASTRUCTURE FIX: {description} resolves {percentage}% of failures",
    'model_switch': "MODEL OPTIMIZATION: Switch to {model} for {cost_reduction}% cost savings", 
    'config_change': "CONFIGURATION FIX: {change} improves reliability by {improvement} pp",
    'prompt_enhancement': "PROMPT ENHANCEMENT: {modification} reduces ambiguity by {improvement}%",
    'tool_configuration': "TOOL OPTIMIZATION: {tool_name} configuration prevents {failure_type}",
    'timeout_adjustment': "TIMEOUT OPTIMIZATION: {timeout_value}s prevents {percentage}% of timeouts",
    'retry_policy': "RETRY ENHANCEMENT: {policy} improves success rate by {improvement}%",
    'validation_addition': "VALIDATION ADDITION: {validator} catches {percentage}% of input errors",
    'fallback_implementation': "FALLBACK STRATEGY: {fallback} maintains {reliability}% uptime",
    'monitoring_enhancement': "MONITORING UPGRADE: {metric} provides {improvement}% faster detection",
}

# Business impact translation - quantified value delivery
IMPACT_MESSAGES = {
    'incident_prevention': "${amount:,} in prevented incidents per month",
    'time_savings': "{hours} engineering hours saved vs reactive debugging",
    'trust_impact': "{percentage}% reduction in customer-facing errors", 
    'cost_reduction': "${amount:,} monthly savings from optimized execution",
    'reliability_improvement': "{percentage}% improvement in SLA compliance",
    'risk_mitigation': "{factor}x reduction in production failure risk",
    'performance_gain': "{percentage}% faster agent response times",
    'scalability_improvement': "{factor}x increase in concurrent user capacity",
    'compliance_enhancement': "{percentage}% improvement in audit compliance",
    'customer_satisfaction': "{percentage}% reduction in support tickets",
}

# Real-time discovery messages for streaming execution
DISCOVERY_MESSAGES = {
    'assumption_found': "DISCOVERED: {assumption_type} violation in scenario {scenario_id}",
    'pattern_emerging': "PATTERN DETECTED: {percentage}% of failures show {pattern_type}",
    'cost_alert': "COST UPDATE: ${current_cost} (${projected_cost} projected)",
    'bottleneck_identified': "BOTTLENECK FOUND: {step_name} causes {percentage}% of delays",
    'error_cluster': "ERROR CLUSTER: {error_type} affects {count} scenarios",
    'success_pattern': "SUCCESS PATTERN: {pattern} achieves {percentage}% reliability",
    'outlier_detected': "OUTLIER DETECTED: Scenario {id} shows unusual {behavior}",
    'threshold_breach': "THRESHOLD BREACH: {metric} exceeds {threshold} limit",
    'optimization_opportunity': "OPTIMIZATION: {opportunity} could save {amount}",
    'validation_failure': "VALIDATION FAILURE: {validator} rejected {percentage}% of inputs",
}

# Progressive insight messages for building confidence over time
PROGRESSIVE_MESSAGES = {
    'partial_analysis': "INTERIM FINDING: {findings} from {completed}/{total} scenarios",
    'confidence_building': "CONFIDENCE: {confidence}% based on {sample_size} results",
    'statistical_significance': "SIGNIFICANCE: p={p_value} with {confidence_interval}% CI",
    'trend_detected': "TREND IDENTIFIED: {trend_description} across {sample_count} samples",
    'hypothesis_validation': "HYPOTHESIS: {hypothesis} supported with {confidence}% confidence",
    'effect_size': "EFFECT SIZE: {effect} represents {interpretation} practical impact",
    'sample_adequacy': "SAMPLE STATUS: {current}/{target} scenarios for {confidence}% confidence",
    'convergence': "CONVERGENCE: Results stabilizing at {value} ± {margin}",
    'power_analysis': "STATISTICAL POWER: {power}% chance of detecting {effect_size} effect",
    'uncertainty_quantification': "UNCERTAINTY: {metric} ± {uncertainty} ({confidence}% CI)",
}

# Infrastructure-first recommendation messaging
INFRASTRUCTURE_FIRST_MESSAGES = {
    'config_before_model': "Try configuration changes before switching models",
    'prompt_before_intelligence': "Enhance prompts before adding model complexity", 
    'validation_before_flexibility': "Add input validation before increasing model freedom",
    'monitoring_before_scaling': "Implement monitoring before increasing throughput",
    'error_handling_before_optimization': "Fix error handling before performance tuning",
    'documentation_before_features': "Update documentation before adding capabilities",
    'testing_before_deployment': "Expand test coverage before production changes",
    'logging_before_debugging': "Enhance logging before investigating failures",
    'automation_before_manual': "Automate fixes before manual intervention",
    'standardization_before_customization': "Standardize configs before custom solutions",
}

# Statistical credibility messages for enterprise trust
STATISTICAL_MESSAGES = {
    'high_confidence': "HIGH CONFIDENCE: p < 0.01, effect size d = {effect_size}",
    'medium_confidence': "MEDIUM CONFIDENCE: p < 0.05, requires validation",
    'low_confidence': "LOW CONFIDENCE: p ≥ 0.05, insufficient evidence",
    'sample_size_adequate': "ADEQUATE SAMPLE: n = {n}, power = {power}%",
    'sample_size_inadequate': "SMALL SAMPLE: n = {n}, results preliminary",
    'effect_size_large': "LARGE EFFECT: Cohen's d = {d}, practically significant",
    'effect_size_medium': "MEDIUM EFFECT: Cohen's d = {d}, meaningful improvement",
    'effect_size_small': "SMALL EFFECT: Cohen's d = {d}, limited practical value",
    'ci_narrow': "PRECISE ESTIMATE: ±{margin}% margin of error",
    'ci_wide': "UNCERTAIN ESTIMATE: ±{margin}% margin, need more data",
}

# Business value proposition messages
VALUE_PROPOSITION_MESSAGES = {
    'proactive_discovery': "PROACTIVE VALUE: {count} issues found BEFORE production",
    'reactive_cost_avoided': "COST AVOIDANCE: ${amount} vs reactive debugging",
    'time_to_resolution': "RAPID RESOLUTION: {minutes} min from problem to fix",
    'reliability_improvement': "RELIABILITY GAIN: {before}% → {after}% success rate",
    'cost_optimization': "COST OPTIMIZATION: ${before} → ${after} per execution",
    'risk_reduction': "RISK REDUCTION: {before}% → {after}% failure rate",
    'scalability_impact': "SCALABILITY: {before} → {after} concurrent users supported",
    'compliance_improvement': "COMPLIANCE: {percentage}% improvement in audit metrics",
    'customer_impact': "CUSTOMER IMPACT: {percentage}% error reduction in production",
    'competitive_advantage': "ADVANTAGE: {factor}x faster issue resolution vs manual testing",
}

# Command-specific messaging templates
COMMAND_MESSAGES = {
    'run': {
        'starting': "INITIATING proactive capability validation",
        'progress': "EXECUTING {current}/{total} scenarios ({percentage}% complete)",
        'discovery': "DISCOVERY: {findings} issues identified",
        'completion': "VALIDATION COMPLETE: {total} scenarios in {duration}",
        'summary': "PROACTIVE DISCOVERY: {issues} capability gaps found BEFORE production",
    },
    'analyze': {
        'starting': "ANALYZING execution patterns and failure origins",
        'decomposition': "DECOMPOSING {count} failures into root causes",
        'funnel': "FUNNEL ANALYSIS: {percentage}% failures at {step} step",
        'patterns': "PATTERN ANALYSIS: {patterns} recurring failure types",
        'completion': "ANALYSIS COMPLETE: {insights} actionable insights generated",
    },
    'recommend': {
        'starting': "GENERATING infrastructure-first improvement recommendations", 
        'prioritization': "PRIORITIZING {count} fixes by impact and effort",
        'infrastructure': "INFRASTRUCTURE FOCUS: {percentage}% fixes are config/prompt changes",
        'model_analysis': "MODEL ANALYSIS: Comparing {count} alternatives", 
        'completion': "RECOMMENDATIONS READY: {count} actionable fixes prioritized",
    },
    'diff': {
        'starting': "VALIDATING assumptions with statistical comparison",
        'comparison': "COMPARING {before} vs {after} configurations",
        'statistical': "STATISTICAL VALIDATION: {test} with p = {p_value}",
        'improvement': "IMPROVEMENT VALIDATED: {metric} improved by {amount}",
        'completion': "VALIDATION COMPLETE: Changes statistically significant",
    },
    'status': {
        'dashboard': "CAPABILITY DASHBOARD: {metrics} tracked over {period}",
        'trends': "TREND ANALYSIS: {direction} trajectory in {metric}",
        'optimization': "OPTIMIZATION OPPORTUNITIES: {count} identified",
        'value': "VALUE DELIVERED: ${amount} in validated improvements",
        'health': "SYSTEM HEALTH: {percentage}% of capabilities above threshold",
    },
}

# Error and warning message templates
ERROR_MESSAGES = {
    'config_invalid': "CONFIGURATION ERROR: {issue} in {file}:{line}",
    'data_insufficient': "INSUFFICIENT DATA: Need {required} samples, have {actual}",
    'statistical_invalid': "STATISTICAL ERROR: {test} requirements not met",
    'assumption_unvalidated': "UNVALIDATED ASSUMPTION: {assumption} needs verification",
    'infrastructure_missing': "INFRASTRUCTURE MISSING: {component} required for {feature}",
    'model_unavailable': "MODEL UNAVAILABLE: {model} not accessible",
    'timeout_exceeded': "TIMEOUT EXCEEDED: {operation} took {duration} > {limit}",
    'validation_failed': "VALIDATION FAILED: {validator} rejected {input}",
    'dependency_missing': "DEPENDENCY MISSING: {dependency} required for {operation}",
    'permission_denied': "ACCESS DENIED: {resource} requires {permission}",
}

# Success and completion messages
SUCCESS_MESSAGES = {
    'validation_passed': "VALIDATION PASSED: All {count} checks successful",
    'improvement_confirmed': "IMPROVEMENT CONFIRMED: {metric} increased by {amount}",
    'assumption_validated': "ASSUMPTION VALIDATED: {assumption} holds with {confidence}% confidence",
    'fix_implemented': "FIX IMPLEMENTED: {fix} resolved {percentage}% of issues",
    'optimization_deployed': "OPTIMIZATION DEPLOYED: {improvement} in {metric}",
    'compliance_achieved': "COMPLIANCE ACHIEVED: {standard} requirements met",
    'threshold_met': "THRESHOLD MET: {metric} exceeds {target} goal",
    'convergence_achieved': "CONVERGENCE ACHIEVED: Results stable at {value}",
    'significance_reached': "SIGNIFICANCE REACHED: p = {p_value} < 0.05",
    'reliability_improved': "RELIABILITY IMPROVED: {before}% → {after}% success rate",
}

# Message formatting helpers
def format_percentage(value: float) -> str:
    """Format percentage with appropriate precision."""
    if value >= 10:
        return f"{value:.1f}%"
    else:
        return f"{value:.2f}%"

def format_currency(amount: float) -> str:
    """Format currency amounts with appropriate scaling."""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.2f}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds:.1f}s"

def format_confidence_interval(mean: float, margin: float, confidence: int = 95) -> str:
    """Format confidence interval for statistical reporting."""
    return f"{mean:.3f} ± {margin:.3f} ({confidence}% CI)"

def format_p_value(p: float) -> str:
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"