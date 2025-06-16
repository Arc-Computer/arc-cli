"""
Data Extraction Utilities

Extracts simulation data from run results and converts it into the format
needed by the generic analyzer. Works with any agent configuration.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import os

from .generic_analyzer import SimulationContext

logger = logging.getLogger(__name__)


class SimulationDataExtractor:
    """Extracts simulation data from various sources for generic analysis."""
    
    def __init__(self):
        pass
    
    def extract_from_run_result(self, run_result) -> SimulationContext:
        """Extract simulation context from a run result object."""
        
        # Load original YAML configuration
        original_yaml = self._load_original_yaml(run_result.config_path)
        
        # Extract scenario data and trajectories from results
        scenario_data = []
        execution_trajectories = []
        failure_data = []
        success_data = []
        
        for result in run_result.results:
            scenario = result.get("scenario", {})
            trajectory = result.get("trajectory", {})
            
            # Add to scenario data
            scenario_data.append(scenario)
            
            # Add to trajectories
            execution_trajectories.append(trajectory)
            
            # Categorize as success or failure
            if trajectory.get("status") == "error":
                failure_info = {
                    "scenario_name": scenario.get("name", "unknown"),
                    "error_message": trajectory.get("error", ""),
                    "task_prompt": scenario.get("task_prompt", ""),
                    "expected_tools": scenario.get("expected_tools", []),
                    "scenario_type": scenario.get("inferred_domain", "unknown"),
                    "complexity": scenario.get("complexity_level", "unknown"),
                    "execution_time": trajectory.get("execution_time_seconds", 0)
                }
                failure_data.append(failure_info)
            else:
                success_info = {
                    "scenario_name": scenario.get("name", "unknown"),
                    "task_prompt": scenario.get("task_prompt", ""),
                    "expected_tools": scenario.get("expected_tools", []),
                    "scenario_type": scenario.get("inferred_domain", "unknown"),
                    "complexity": scenario.get("complexity_level", "unknown"),
                    "execution_time": trajectory.get("execution_time_seconds", 0)
                }
                success_data.append(success_info)
        
        # Create performance metrics
        performance_metrics = {
            "reliability_score": run_result.reliability_score,
            "total_scenarios": len(run_result.results),
            "success_count": len(success_data),
            "failure_count": len(failure_data),
            "avg_execution_time": sum(r.get("trajectory", {}).get("execution_time_seconds", 0) for r in run_result.results) / len(run_result.results) if run_result.results else 0,
            "total_cost": run_result.total_cost,
            "run_id": run_result.run_id,
            "timestamp": run_result.timestamp
        }
        
        return SimulationContext(
            original_yaml=original_yaml,
            scenario_data=scenario_data,
            execution_trajectories=execution_trajectories,
            failure_data=failure_data,
            success_data=success_data,
            performance_metrics=performance_metrics
        )
    
    def _load_original_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load the original YAML configuration."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        # Fallback to JSON parsing if PyYAML not available
                        f.seek(0)
                        content = f.read()
                        # Try to convert simple YAML to JSON-like format
                        return self._simple_yaml_parse(content)
            else:
                logger.warning(f"Config file not found: {config_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return {}
    
    def _simple_yaml_parse(self, content: str) -> Dict[str, Any]:
        """Simple YAML parser fallback when PyYAML is not available."""
        result = {}
        current_key = None
        current_value = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line and not line.startswith(' '):
                # Save previous key if exists
                if current_key:
                    if len(current_value) == 1:
                        result[current_key] = current_value[0]
                    else:
                        result[current_key] = current_value
                
                # Parse new key
                key, value = line.split(':', 1)
                current_key = key.strip()
                current_value = [value.strip()] if value.strip() else []
                
            elif line.startswith('- ') and current_key:
                # List item
                current_value.append(line[2:].strip())
            elif line.startswith(' ') and current_key:
                # Continuation or nested content
                current_value.append(line.strip())
        
        # Don't forget the last key
        if current_key:
            if len(current_value) == 1:
                result[current_key] = current_value[0]
            else:
                result[current_key] = current_value
        
        return result
    
    def extract_from_analysis_data(self, analysis: Dict[str, Any], config_path: str) -> SimulationContext:
        """Extract simulation context from existing analysis data."""
        
        # Load original YAML
        original_yaml = self._load_original_yaml(config_path)
        
        # Extract data from analysis structure
        failure_data = []
        success_data = []
        scenario_data = []
        execution_trajectories = []
        
        # Process failure clusters if available
        clusters = analysis.get("clusters", [])
        for cluster in clusters:
            failures = cluster.get("failures", [])
            for failure in failures:
                failure_data.append({
                    "scenario_name": failure.get("scenario_name", "unknown"),
                    "error_message": failure.get("error_message", ""),
                    "task_prompt": failure.get("task_prompt", ""),
                    "expected_tools": failure.get("expected_tools", []),
                    "scenario_type": failure.get("scenario_type", "unknown"),
                    "complexity": failure.get("complexity", "unknown"),
                    "execution_time": failure.get("execution_time", 0)
                })
                
                # Create corresponding scenario and trajectory data
                scenario_data.append({
                    "name": failure.get("scenario_name", "unknown"),
                    "task_prompt": failure.get("task_prompt", ""),
                    "expected_tools": failure.get("expected_tools", []),
                    "inferred_domain": failure.get("scenario_type", "unknown"),
                    "complexity_level": failure.get("complexity", "unknown")
                })
                
                execution_trajectories.append(failure.get("raw_trajectory", {}))
        
        # Create performance metrics from analysis
        performance_metrics = {
            "reliability_score": 1.0 - analysis.get("failure_rate", 0),
            "total_scenarios": analysis.get("total_failures", 0),
            "success_count": 0,  # Not available in current analysis
            "failure_count": analysis.get("total_failures", 0),
            "run_id": analysis.get("run_id", "unknown"),
            "database_connected": analysis.get("database_connected", False)
        }
        
        return SimulationContext(
            original_yaml=original_yaml,
            scenario_data=scenario_data,
            execution_trajectories=execution_trajectories,
            failure_data=failure_data,
            success_data=success_data,
            performance_metrics=performance_metrics
        )
    
    def enrich_with_historical_data(self, context: SimulationContext, historical_patterns: Optional[Dict] = None) -> SimulationContext:
        """Enrich simulation context with historical data if available."""
        if not historical_patterns:
            return context
        
        # Add historical context to performance metrics
        context.performance_metrics.update({
            "historical_avg_reliability": historical_patterns.get("avg_reliability", 0),
            "historical_patterns": historical_patterns.get("patterns", []),
            "historical_period_days": historical_patterns.get("period_days", 0)
        })
        
        return context 