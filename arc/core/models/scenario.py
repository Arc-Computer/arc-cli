"""Scenario definition model for Arc-Eval."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib
import json

__all__: list[str] = ["Scenario"]


class Scenario(BaseModel):
    """Test scenario for agent evaluation."""

    # Core fields
    id: str = Field(default="", description="Unique scenario identifier")
    name: str = Field(default="", description="Human-readable scenario name")
    task_prompt: str = Field(..., description="The task prompt for the agent")
    
    # Expected behavior
    expected_tools: List[str] = Field(default_factory=list, description="Tools expected to be used")
    expected_steps: List[str] = Field(default_factory=list, description="Expected execution steps")
    
    # Failure information
    potential_failure_mode: str = Field(default="", description="Expected failure mode")
    expected_error: str = Field(default="", description="Expected error type")
    
    # Metadata
    complexity_level: str = Field(default="medium", description="Scenario complexity: low, medium, high")
    inferred_domain: str = Field(default="general", description="Domain category")
    generation_strategy: str = Field(default="llm", description="How scenario was generated")
    
    # Quality and pattern info
    pattern_id: Optional[str] = Field(default=None, description="Source pattern ID if pattern-based")
    quality_score: Optional[float] = Field(default=None, description="Quality score")
    
    # Success criteria
    success_criteria: Dict[str, Any] = Field(default_factory=dict, description="Success evaluation criteria")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, **data):
        """Initialize scenario with auto-generated ID if not provided"""
        if 'id' not in data or not data['id']:
            # Generate ID from task prompt
            prompt_hash = hashlib.md5(data.get('task_prompt', '').encode()).hexdigest()[:8]
            data['id'] = f"scenario_{prompt_hash}"
        
        if 'name' not in data or not data['name']:
            # Generate name from task prompt
            data['name'] = data.get('task_prompt', '')[:50] + "..."
            
        super().__init__(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = self.model_dump()
        # Convert datetime to ISO format
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create scenario from dictionary"""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def to_sandbox_format(self) -> Dict[str, Any]:
        """Convert to format expected by sandbox"""
        return {
            "id": self.id,
            "name": self.name,
            "task_prompt": self.task_prompt,
            "expected_tools": self.expected_tools,
            "evaluation_criteria": {
                "expected_tools": self.expected_tools,
                "expected_steps": self.expected_steps,
                "success_criteria": self.success_criteria,
                "pattern_id": self.pattern_id
            },
            "metadata": {
                "complexity": self.complexity_level,
                "domain": self.inferred_domain,
                "generation_strategy": self.generation_strategy,
                "quality_score": self.quality_score,
                "expected_error": self.expected_error or self.potential_failure_mode
            }
        }
