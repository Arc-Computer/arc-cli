"""
Fast pattern selection for scenario generation.
Selects the most relevant failure patterns based on agent configuration.
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
import asyncio

from .failure_patterns import PatternLibrary, FailurePattern
from .assumption_extractor import AgentAssumptions


@dataclass
class PatternSelectionResult:
    """Result of pattern selection process."""
    selected_patterns: List[FailurePattern]
    selection_reasoning: str
    selection_method: str  # "llm", "heuristic", "random"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_patterns": [p.to_dict() for p in self.selected_patterns],
            "selection_reasoning": self.selection_reasoning,
            "selection_method": self.selection_method
        }


class PatternSelector:
    """Stage A: Selects relevant failure patterns based on agent config."""
    
    def __init__(
        self,
        pattern_library: PatternLibrary,
        api_key: Optional[str] = None,
        use_llm: bool = True
    ):
        """Initialize pattern selector.
        
        Args:
            pattern_library: Library of failure patterns
            api_key: OpenRouter API key for LLM selection
            use_llm: Whether to use LLM for intelligent selection
        """
        self.library = pattern_library
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_llm = use_llm and bool(self.api_key)
        
        # Model for pattern selection (fast and cheap)
        self.selector_model = os.getenv("ARC_SELECTOR_MODEL", "openai/gpt-4.1-mini")
        
        # Track recently used patterns for diversity
        self._recent_patterns: List[str] = []
        self._selection_history: List[PatternSelectionResult] = []
    
    async def select_patterns(
        self,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        count: int = 3,
        prioritize_assumptions: bool = True
    ) -> PatternSelectionResult:
        """Select the most relevant patterns for the agent.
        
        Args:
            assumptions: Extracted agent assumptions
            agent_config: Full agent configuration
            count: Number of patterns to select
            prioritize_assumptions: Whether to prioritize assumption violations
            
        Returns:
            Pattern selection result
        """
        # Get candidate patterns based on tools
        tool_patterns = self.library.get_patterns_for_tools(assumptions.tools)
        
        if not tool_patterns:
            # No tool-specific patterns, use all patterns
            tool_patterns = self.library.get_all_patterns()
        
        # Filter by assumptions if requested
        if prioritize_assumptions:
            assumption_patterns = self._filter_by_assumptions(tool_patterns, assumptions)
            if assumption_patterns:
                tool_patterns = assumption_patterns
        
        # Select patterns using appropriate method
        if self.use_llm and len(tool_patterns) > count * 2:
            # Use LLM for intelligent selection when we have many candidates
            try:
                result = await self._llm_selection(
                    tool_patterns, assumptions, agent_config, count
                )
                return result
            except Exception as e:
                print(f"LLM selection failed, falling back to heuristic: {e}")
        
        # Fall back to heuristic selection
        return self._heuristic_selection(tool_patterns, assumptions, count)
    
    def _filter_by_assumptions(
        self,
        patterns: List[FailurePattern],
        assumptions: AgentAssumptions
    ) -> List[FailurePattern]:
        """Filter patterns that test specific assumptions."""
        filtered = []
        
        for pattern in patterns:
            # Check if pattern tests any extracted assumptions
            pattern_tests_assumption = False
            
            # Currency assumptions
            if assumptions.currencies and pattern.category == "calculation":
                pattern_tests_assumption = True
            
            # Data format assumptions
            if assumptions.data_formats and pattern.category == "data":
                if any(fmt in pattern.description.lower() for fmt in assumptions.data_formats):
                    pattern_tests_assumption = True
            
            # API/infrastructure assumptions
            if (assumptions.api_versions or assumptions.rate_limits) and pattern.category == "infrastructure":
                pattern_tests_assumption = True
            
            # Timeout assumptions
            if assumptions.timeouts and "timeout" in pattern.id:
                pattern_tests_assumption = True
            
            # Error handling assumptions
            if assumptions.error_handling and pattern.category in ["logic", "infrastructure"]:
                pattern_tests_assumption = True
            
            if pattern_tests_assumption:
                filtered.append(pattern)
        
        return filtered
    
    def _heuristic_selection(
        self,
        candidates: List[FailurePattern],
        assumptions: AgentAssumptions,
        count: int
    ) -> PatternSelectionResult:
        """Select patterns using heuristic rules."""
        # Use library's diversity selection
        selected = self.library.select_diverse_patterns(
            candidates,
            count=count,
            prioritize_severity=True
        )
        
        # Build reasoning
        reasoning_parts = []
        
        if assumptions.currencies:
            reasoning_parts.append(f"Testing currency assumptions: {', '.join(assumptions.currencies)}")
        
        if assumptions.data_formats:
            reasoning_parts.append(f"Testing data formats: {', '.join(assumptions.data_formats)}")
        
        reasoning_parts.append(f"Selected {len(selected)} diverse patterns across categories")
        
        categories = set(p.category for p in selected)
        reasoning_parts.append(f"Categories covered: {', '.join(categories)}")
        
        return PatternSelectionResult(
            selected_patterns=selected,
            selection_reasoning=" | ".join(reasoning_parts),
            selection_method="heuristic"
        )
    
    async def _llm_selection(
        self,
        candidates: List[FailurePattern],
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        count: int
    ) -> PatternSelectionResult:
        """Use LLM to intelligently select patterns."""
        # Prepare prompt
        prompt = self._create_selection_prompt(
            candidates, assumptions, agent_config, count
        )
        
        # Call LLM
        selected_ids = await self._call_llm_for_selection(prompt)
        
        # Map IDs back to patterns
        selected_patterns = []
        for pattern_id in selected_ids[:count]:
            pattern = self.library.get_pattern(pattern_id)
            if pattern and pattern in candidates:
                selected_patterns.append(pattern)
        
        # Fill with heuristic selection if needed
        if len(selected_patterns) < count:
            remaining = [p for p in candidates if p not in selected_patterns]
            additional = self.library.select_diverse_patterns(
                remaining,
                count=count - len(selected_patterns),
                prioritize_severity=True
            )
            selected_patterns.extend(additional)
        
        return PatternSelectionResult(
            selected_patterns=selected_patterns,
            selection_reasoning=f"LLM selected patterns based on agent capabilities and assumptions",
            selection_method="llm"
        )
    
    def _create_selection_prompt(
        self,
        patterns: List[FailurePattern],
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        count: int
    ) -> str:
        """Create prompt for LLM pattern selection."""
        # Format patterns for selection
        pattern_list = []
        for p in patterns[:30]:  # Limit to prevent prompt overflow
            pattern_list.append(
                f"- {p.id} ({p.category}/{p.severity}): {p.title}\n"
                f"  Description: {p.description[:100]}..."
            )
        
        # Format assumptions
        assumption_summary = []
        if assumptions.currencies:
            assumption_summary.append(f"Currencies: {', '.join(list(assumptions.currencies)[:5])}")
        if assumptions.data_formats:
            assumption_summary.append(f"Data formats: {', '.join(list(assumptions.data_formats)[:5])}")
        if assumptions.capabilities:
            assumption_summary.append(f"Capabilities: {', '.join(list(assumptions.capabilities)[:5])}")
        
        prompt = f"""You are an expert at selecting failure patterns for testing AI agents.

AGENT CONFIGURATION:
- Model: {agent_config.get('model', 'unknown')}
- Tools: {', '.join(assumptions.tools[:10])}
- Assumptions: {' | '.join(assumption_summary)}

AVAILABLE PATTERNS ({len(patterns)} total):
{chr(10).join(pattern_list)}

Select the {count} most relevant patterns that would:
1. Test the agent's specific assumptions and capabilities
2. Cover high-severity failure modes
3. Provide diverse testing across different error categories
4. Focus on patterns likely to occur with the given tools

Return ONLY a JSON array of pattern IDs, like: ["pattern_id_1", "pattern_id_2", "pattern_id_3"]
"""
        
        return prompt
    
    async def _call_llm_for_selection(self, prompt: str) -> List[str]:
        """Call LLM to select pattern IDs."""
        if not self.api_key:
            raise ValueError("No API key available for LLM selection")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://arc-eval.com",
            "X-Title": "Arc Evaluation Platform"
        }
        
        payload = {
            "model": self.selector_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pattern selector for testing AI systems. Always respond with valid JSON arrays containing pattern IDs."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 200
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
                    
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'patterns' in data:
                        return data['patterns']
                    else:
                        return []
                except json.JSONDecodeError:
                    # Try to extract pattern IDs with regex as fallback
                    import re
                    pattern_ids = re.findall(r'"([^"]+)"', content)
                    return pattern_ids
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern selection."""
        if not self._selection_history:
            return {"total_selections": 0}
        
        methods = [r.selection_method for r in self._selection_history]
        method_counts = {
            "llm": methods.count("llm"),
            "heuristic": methods.count("heuristic"),
            "random": methods.count("random")
        }
        
        all_patterns = []
        for result in self._selection_history:
            all_patterns.extend([p.id for p in result.selected_patterns])
        
        pattern_frequency = {}
        for pid in all_patterns:
            pattern_frequency[pid] = pattern_frequency.get(pid, 0) + 1
        
        return {
            "total_selections": len(self._selection_history),
            "selection_methods": method_counts,
            "unique_patterns_used": len(set(all_patterns)),
            "most_frequent_patterns": sorted(
                pattern_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }