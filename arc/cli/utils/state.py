"""CLI state management for storing runs and analysis results."""

import json
import os
import fcntl
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager


@dataclass
class RunResult:
    """Results from an arc run command."""
    run_id: str
    config_path: str
    timestamp: datetime
    scenario_count: int
    success_count: int
    failure_count: int
    reliability_score: float
    execution_time: float
    total_cost: float
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None
    
    @property
    def reliability_percentage(self) -> float:
        """Get reliability as percentage."""
        return self.reliability_score * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResult':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CLIState:
    """Manages Arc CLI state and run history."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """Initialize CLI state manager.
        
        Args:
            state_dir: Directory for storing state (defaults to ~/.arc)
        """
        self.state_dir = state_dir or Path.home() / ".arc"
        self.runs_dir = self.state_dir / "runs"
        self.config_file = self.state_dir / "config.json"
        
        # Lock file for concurrent access protection
        self.lock_file = self.state_dir / ".lock"
        
        # Create directories if they don't exist
        self.state_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
        # Load or initialize config
        self.config = self._load_config()
    
    @contextmanager
    def _file_lock(self, timeout: float = 5.0):
        """Acquire file lock for concurrent access protection."""
        lock_acquired = False
        start_time = time.time()
        
        # Create lock file if it doesn't exist
        self.lock_file.touch(exist_ok=True)
        
        with open(self.lock_file, 'w') as lock_fd:
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except IOError:
                    if time.time() - start_time > timeout:
                        raise RuntimeError("Could not acquire file lock - another Arc process may be running")
                    time.sleep(0.1)
            
            try:
                yield
            finally:
                if lock_acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration."""
        # Define expected schema with defaults
        schema = {
            "last_run_id": (type(None), str),
            "default_scenario_count": (int,),
            "cost_warning_threshold": (float,),
            "use_modal": (bool,),
            "max_concurrent_runs": (int,),
        }
        
        defaults = {
            "last_run_id": None,
            "default_scenario_count": 50,
            "cost_warning_threshold": 0.10,
            "use_modal": True,
            "max_concurrent_runs": 5,
        }
        
        validated = {}
        
        for key, expected_types in schema.items():
            if key in config:
                value = config[key]
                if value is not None and not isinstance(value, expected_types):
                    # Try to convert if possible
                    if int in expected_types and isinstance(value, (str, float)):
                        try:
                            value = int(value)
                        except ValueError:
                            value = defaults[key]
                    elif float in expected_types and isinstance(value, (str, int)):
                        try:
                            value = float(value)
                        except ValueError:
                            value = defaults[key]
                    else:
                        value = defaults[key]
                validated[key] = value
            else:
                validated[key] = defaults[key]
        
        # Validate ranges
        if validated["default_scenario_count"] < 1:
            validated["default_scenario_count"] = 50
        if validated["cost_warning_threshold"] < 0:
            validated["cost_warning_threshold"] = 0.10
        if validated["max_concurrent_runs"] < 1:
            validated["max_concurrent_runs"] = 5
            
        return validated
    
    def _load_config(self) -> Dict[str, Any]:
        """Load CLI configuration."""
        if self.config_file.exists():
            with self._file_lock():
                try:
                    with open(self.config_file, 'r') as f:
                        raw_config = json.load(f)
                    return self._validate_config(raw_config)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Config file is corrupted, return defaults
                    return self._validate_config({})
        return self._validate_config({})
    
    def _save_config(self) -> None:
        """Save CLI configuration."""
        with self._file_lock():
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def save_run(self, result: RunResult) -> Path:
        """Save run results.
        
        Args:
            result: Run result to save
            
        Returns:
            Path to saved run directory
        """
        with self._file_lock():
            # Create run directory
            run_dir = self.runs_dir / result.run_id
            run_dir.mkdir(exist_ok=True)
            
            # Save run metadata
            with open(run_dir / "result.json", 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Save scenarios
            with open(run_dir / "scenarios.json", 'w') as f:
                json.dump(result.scenarios, f, indent=2)
            
            # Save results
            with open(run_dir / "results.json", 'w') as f:
                json.dump(result.results, f, indent=2)
            
            # Update last run
            self.config["last_run_id"] = result.run_id
            self._save_config()
            
            return run_dir
    
    def get_run(self, run_id: Optional[str] = None) -> Optional[RunResult]:
        """Get run results by ID or last run.
        
        Args:
            run_id: Specific run ID or None for last run
            
        Returns:
            Run result or None if not found
        """
        if run_id is None:
            run_id = self.config.get("last_run_id")
            if run_id is None:
                return None
        
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None
        
        # Load run result
        with open(run_dir / "result.json", 'r') as f:
            data = json.load(f)
        
        result = RunResult.from_dict(data)
        
        # Load scenarios if not in result
        if not result.scenarios and (run_dir / "scenarios.json").exists():
            with open(run_dir / "scenarios.json", 'r') as f:
                result.scenarios = json.load(f)
        
        # Load results if not in result
        if not result.results and (run_dir / "results.json").exists():
            with open(run_dir / "results.json", 'r') as f:
                result.results = json.load(f)
        
        # Load analysis if exists
        if (run_dir / "analysis.json").exists():
            with open(run_dir / "analysis.json", 'r') as f:
                result.analysis = json.load(f)
        
        # Load recommendations if exists
        if (run_dir / "recommendations.json").exists():
            with open(run_dir / "recommendations.json", 'r') as f:
                result.recommendations = json.load(f)
        
        return result
    
    def save_analysis(self, run_id: str, analysis: Dict[str, Any]) -> None:
        """Save analysis results for a run."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")
        
        with open(run_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def save_recommendations(self, run_id: str, recommendations: Dict[str, Any]) -> None:
        """Save recommendations for a run."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")
        
        with open(run_dir / "recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)
    
    def save_diff(self, diff_id: str, diff_result: Dict[str, Any]) -> None:
        """Save A/B test diff results.
        
        Args:
            diff_id: Diff session ID
            diff_result: Diff comparison data
        """
        diffs_dir = self.state_dir / "diffs"
        diffs_dir.mkdir(exist_ok=True)
        
        diff_file = diffs_dir / f"{diff_id}.json"
        with open(diff_file, 'w') as f:
            json.dump(diff_result, f, indent=2, default=str)
    
    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of run summaries
        """
        runs = []
        
        # Get all run directories
        run_dirs = sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        
        for run_dir in run_dirs[:limit]:
            try:
                with open(run_dir / "result.json", 'r') as f:
                    data = json.load(f)
                
                runs.append({
                    "run_id": data["run_id"],
                    "config_path": data["config_path"],
                    "timestamp": data["timestamp"],
                    "reliability_score": data["reliability_score"],
                    "scenario_count": data["scenario_count"],
                    "total_cost": data["total_cost"]
                })
            except Exception:
                # Skip corrupted runs
                continue
        
        return runs
    
    def get_total_cost(self, days: int = 30) -> float:
        """Get total cost for recent runs.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Total cost in dollars
        """
        total = 0.0
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            if run_dir.stat().st_mtime < cutoff:
                continue
            
            try:
                with open(run_dir / "result.json", 'r') as f:
                    data = json.load(f)
                total += data.get("total_cost", 0)
            except Exception:
                continue
        
        return total