"""
Scenario instantiation from selected patterns.
Converts abstract patterns into concrete test scenarios.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
import asyncio
from dataclasses import dataclass

from ..core.models.scenario import Scenario
from .failure_patterns import FailurePattern
from .assumption_extractor import AgentAssumptions
from .pattern_adapter import PatternAdapter, ScenarioTemplate


@dataclass
class InstantiationResult:
    """Result of scenario instantiation."""
    scenarios: List[Scenario]
    generation_method: str  # "llm", "pattern_adapter", "hybrid"
    generation_time: float
    total_generated: int
    quality_filtered: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "generation_method": self.generation_method,
            "generation_time": self.generation_time,
            "total_generated": self.total_generated,
            "quality_filtered": self.quality_filtered
        }


class ScenarioInstantiator:
    """Stage B: Instantiates concrete scenarios from patterns."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_llm: bool = True,
        quality_threshold: float = 3.0
    ):
        """Initialize scenario instantiator.
        
        Args:
            api_key: OpenRouter API key for LLM generation
            use_llm: Whether to use LLM for scenario generation
            quality_threshold: Minimum quality score for scenarios
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_llm = use_llm and bool(self.api_key)
        self.quality_threshold = quality_threshold
        
        # Model for scenario instantiation (better quality)
        self.instantiator_model = os.getenv("ARC_INSTANTIATOR_MODEL", "openai/gpt-4.1")
        
        # Pattern adapter for non-LLM generation
        self.pattern_adapter = PatternAdapter()
        
        # Few-shot examples for LLM
        self.few_shot_examples = self._load_few_shot_examples()
    
    async def instantiate_scenarios(
        self,
        patterns: List[FailurePattern],
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        scenarios_per_pattern: int = 5,
        use_hybrid: bool = True
    ) -> InstantiationResult:
        """Generate concrete scenarios from patterns.
        
        Args:
            patterns: Selected failure patterns
            assumptions: Agent assumptions
            agent_config: Full agent configuration
            scenarios_per_pattern: Number of scenarios per pattern
            use_hybrid: Whether to use both LLM and pattern adapter
            
        Returns:
            Instantiation result with generated scenarios
        """
        start_time = datetime.now()
        all_scenarios = []
        
        for pattern in patterns:
            if use_hybrid and self.use_llm:
                # Generate half with LLM, half with pattern adapter
                llm_count = scenarios_per_pattern // 2
                adapter_count = scenarios_per_pattern - llm_count
                
                # LLM generation
                if llm_count > 0:
                    try:
                        llm_scenarios = await self._llm_instantiation(
                            pattern, assumptions, agent_config, llm_count
                        )
                        all_scenarios.extend(llm_scenarios)
                    except Exception as e:
                        print(f"LLM instantiation failed for {pattern.id}: {e}")
                        adapter_count = scenarios_per_pattern  # Fall back to all adapter
                
                # Pattern adapter generation
                if adapter_count > 0:
                    # Extract tool names if tools are dicts
                    tools = agent_config.get('tools', [])
                    tool_names = [t.get('name', str(t)) if isinstance(t, dict) else t for t in tools]
                    adapter_scenarios = self._adapter_instantiation(
                        pattern, assumptions, tool_names, adapter_count
                    )
                    all_scenarios.extend(adapter_scenarios)
            
            elif self.use_llm:
                # Pure LLM generation
                try:
                    llm_scenarios = await self._llm_instantiation(
                        pattern, assumptions, agent_config, scenarios_per_pattern
                    )
                    all_scenarios.extend(llm_scenarios)
                except Exception as e:
                    print(f"LLM instantiation failed, using adapter: {e}")
                    # Extract tool names if tools are dicts
                    tools = agent_config.get('tools', [])
                    tool_names = [t.get('name', str(t)) if isinstance(t, dict) else t for t in tools]
                    adapter_scenarios = self._adapter_instantiation(
                        pattern, assumptions, tool_names, scenarios_per_pattern
                    )
                    all_scenarios.extend(adapter_scenarios)
            
            else:
                # Pure adapter generation
                # Extract tool names if tools are dicts
                tools = agent_config.get('tools', [])
                tool_names = [t.get('name', str(t)) if isinstance(t, dict) else t for t in tools]
                adapter_scenarios = self._adapter_instantiation(
                    pattern, assumptions, tool_names, scenarios_per_pattern
                )
                all_scenarios.extend(adapter_scenarios)
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Determine generation method
        if use_hybrid and self.use_llm:
            method = "hybrid"
        elif self.use_llm:
            method = "llm"
        else:
            method = "pattern_adapter"
        
        return InstantiationResult(
            scenarios=all_scenarios,
            generation_method=method,
            generation_time=generation_time,
            total_generated=len(all_scenarios),
            quality_filtered=0  # Quality filtering will be done separately
        )
    
    def _adapter_instantiation(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[Scenario]:
        """Generate scenarios using pattern adapter."""
        templates = self.pattern_adapter.adapt_pattern(
            pattern, assumptions, tools, count
        )
        
        scenarios = []
        for i, template in enumerate(templates):
            scenario = Scenario(
                name=f"{pattern.category}_{pattern.id}_{i+1}",
                task_prompt=template.task_prompt,
                expected_tools=template.expected_tools,
                potential_failure_mode=template.expected_error,
                expected_error=pattern.severity,
                complexity_level="medium",
                inferred_domain=pattern.category,
                generation_strategy="pattern_adapter",
                pattern_id=pattern.id,
                success_criteria={
                    "must_handle_error": True,
                    "expected_behavior": template.expected_error
                }
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _llm_instantiation(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        count: int
    ) -> List[Scenario]:
        """Generate scenarios using LLM."""
        prompt = self._create_instantiation_prompt(
            pattern, assumptions, agent_config, count
        )
        
        scenario_data = await self._call_llm_for_instantiation(prompt)
        
        scenarios = []
        for i, data in enumerate(scenario_data[:count]):
            scenario = Scenario(
                name=f"{pattern.category}_{pattern.id}_llm_{i+1}",
                task_prompt=data['task_prompt'],
                expected_tools=data.get('expected_tools', []),
                potential_failure_mode=data.get('expected_error', ''),
                expected_error=pattern.severity,
                complexity_level=data.get('complexity_level', 'medium'),
                inferred_domain=pattern.category,
                generation_strategy="llm",
                pattern_id=pattern.id,
                success_criteria={
                    "must_handle_error": True,
                    "expected_behavior": data.get('expected_error', '')
                }
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _create_instantiation_prompt(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        count: int
    ) -> str:
        """Create prompt for scenario instantiation."""
        # Format assumptions
        assumption_details = []
        if assumptions.currencies:
            assumption_details.append(f"Currencies: {', '.join(list(assumptions.currencies)[:5])}")
        if assumptions.data_formats:
            assumption_details.append(f"Data formats: {', '.join(list(assumptions.data_formats)[:5])}")
        if assumptions.timeouts:
            assumption_details.append(f"Timeouts: {json.dumps(dict(list(assumptions.timeouts.items())[:3]))}")
        if assumptions.rate_limits:
            assumption_details.append(f"Rate limits: {json.dumps(dict(list(assumptions.rate_limits.items())[:3]))}")
        
        # Format few-shot examples
        examples_text = self._format_few_shot_examples()
        
        prompt = f"""You are an expert at creating test scenarios that trigger specific failure patterns in AI agents.

PATTERN TO INSTANTIATE:
ID: {pattern.id}
Title: {pattern.title}
Category: {pattern.category}
Description: {pattern.description}
Trigger Conditions: {json.dumps(pattern.trigger_conditions)}
Expected Error: {pattern.expected_error}

AGENT CONFIGURATION:
Tools Available: {json.dumps(assumptions.tools[:10])}
Assumptions: {' | '.join(assumption_details)}
Model: {agent_config.get('model', 'unknown')}

{examples_text}

CRITICAL REQUIREMENTS for high-quality scenarios:

1. Task Prompt MUST include edge case keywords (at least 2-3):
   - Required: null, empty, invalid, missing, extreme, zero, negative, infinity, timeout, expired, malformed, non-existent, boundary, overflow, special character
   - Example: "Search for products with null category in empty database with timeout of -1 seconds"

2. Expected Error MUST:
   - Start with action words: "will fail", "causes", "triggers", "results in", "leads to"
   - Include technical details: HTTP codes (404, 500, 503), "timeout", "rate limit", "validation error"
   - Be at least 50 characters long
   - Be specific about how the pattern manifests

3. Tools should match the agent's available tools when possible

4. Each scenario should be unique and test different aspects of the pattern

Generate {count} diverse test scenarios. Return a JSON object:
{{
  "scenarios": [
    {{
      "task_prompt": "Specific task with edge cases that triggers the pattern",
      "expected_tools": ["tool1", "tool2"],
      "expected_error": "The operation will fail with [specific error] because [technical reason]",
      "complexity_level": "simple|medium|complex"
    }}
  ]
}}"""
        
        return prompt
    
    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for the prompt."""
        if not self.few_shot_examples:
            return ""
        
        lines = ["FEW-SHOT EXAMPLES:"]
        
        for i, example in enumerate(self.few_shot_examples[:3], 1):
            lines.append(f"\nExample {i} - Pattern: {example['pattern']['title']}")
            lines.append(f"Task: {example['scenario']['task_prompt']}")
            lines.append(f"Tools: {', '.join(example['scenario']['expected_tools'])}")
            lines.append(f"Error: {example['scenario']['expected_error']}")
            lines.append(f"Complexity: {example['scenario']['complexity_level']}")
        
        lines.append("")  # Empty line after examples
        
        return '\n'.join(lines)
    
    async def _call_llm_for_instantiation(self, prompt: str) -> List[Dict[str, Any]]:
        """Call LLM to generate scenarios."""
        if not self.api_key:
            raise ValueError("No API key available for LLM instantiation")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://arc-eval.com",
            "X-Title": "Arc Evaluation Platform"
        }
        
        payload = {
            "model": self.instantiator_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a scenario generator for testing AI systems. Create realistic, edge-case scenarios that would trigger specific failures."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                
                result = await response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    # Handle markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    data = json.loads(content.strip())
                    return data.get('scenarios', [])
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse LLM response: {e}")
                    return []
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Load high-quality few-shot examples."""
        return [
            {
                "pattern": {
                    "id": "precision_loss",
                    "category": "calculation",
                    "title": "Floating Point Precision Loss"
                },
                "scenario": {
                    "task_prompt": "Convert 0.001 USD to EUR at rate 0.85, then to JPY at rate 149.50, then back to USD at inverse rates. Calculate the exact difference from the original 0.001 USD including all decimal places.",
                    "expected_tools": ["currency_converter", "calculator"],
                    "expected_error": "The multi-step currency conversion will accumulate floating-point rounding errors, resulting in a final value that differs from 0.001 USD by approximately 0.000001-0.00001 USD, demonstrating precision loss in financial calculations.",
                    "complexity_level": "complex"
                }
            },
            {
                "pattern": {
                    "id": "timeout_api_delay",
                    "category": "infrastructure",
                    "title": "API Timeout"
                },
                "scenario": {
                    "task_prompt": "Fetch real-time weather data for all 999999 coordinates in a 1000x1000 grid covering the entire Pacific Ocean, with timeout set to -30 seconds.",
                    "expected_tools": ["weather_api", "coordinate_generator"],
                    "expected_error": "The request will fail immediately because the negative timeout value (-30 seconds) is invalid, causing a configuration error. Even with a valid timeout, fetching data for 999999 locations would exceed any reasonable timeout limit, resulting in a 504 Gateway Timeout.",
                    "complexity_level": "medium"
                }
            },
            {
                "pattern": {
                    "id": "empty_results",
                    "category": "data",
                    "title": "Empty Result Set"
                },
                "scenario": {
                    "task_prompt": "Search for products where price is null AND category is empty string AND name contains only special characters '!@#$%^&*()' in the non-existent database 'undefined_catalog'.",
                    "expected_tools": ["database_search", "data_validator"],
                    "expected_error": "The query will return an empty result set because the combination of null price, empty category, and special-character-only names is logically impossible in any valid product database. Additionally, the database 'undefined_catalog' doesn't exist, causing a secondary failure.",
                    "complexity_level": "simple"
                }
            }
        ]