"""
Adapts failure patterns to concrete scenarios based on agent configuration.
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from .failure_patterns import FailurePattern
from .assumption_extractor import AgentAssumptions


@dataclass
class ScenarioTemplate:
    """Template for generating scenarios from patterns."""
    task_prompt: str
    expected_tools: List[str]
    expected_error: str
    metadata: Dict[str, Any]
    
    def to_scenario_dict(self) -> Dict[str, Any]:
        """Convert to scenario dictionary format."""
        return {
            "task_prompt": self.task_prompt,
            "expected_tools": self.expected_tools,
            "expected_error": self.expected_error,
            "metadata": self.metadata
        }


class PatternAdapter:
    """Adapts failure patterns to concrete scenarios."""
    
    def __init__(self):
        """Initialize the pattern adapter."""
        # Edge case keywords for high-quality scenarios
        self.edge_case_keywords = [
            "null", "empty", "invalid", "missing", "malformed", "corrupted",
            "timeout", "expired", "unauthorized", "forbidden", "limit", "boundary",
            "overflow", "underflow", "negative", "zero", "extreme", "infinity",
            "special character", "unicode", "non-existent", "ambiguous"
        ]
        
        # Pattern-specific adaptation strategies
        self.adaptation_strategies = {
            "precision_loss": self._adapt_precision_loss,
            "malformed_response": self._adapt_malformed_response,
            "timeout_api_delay": self._adapt_timeout,
            "ambiguous_request": self._adapt_ambiguous_request,
            "empty_results": self._adapt_empty_results,
            "rate_limiting": self._adapt_rate_limiting,
            "authentication_error": self._adapt_auth_error,
            "resource_exhaustion": self._adapt_resource_exhaustion
        }
    
    def adapt_pattern(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int = 1
    ) -> List[ScenarioTemplate]:
        """Adapt a failure pattern to concrete scenarios.
        
        Args:
            pattern: The failure pattern to adapt
            assumptions: Agent assumptions extracted from config
            tools: Available tools for the agent
            count: Number of scenarios to generate
            
        Returns:
            List of scenario templates
        """
        # Use pattern-specific strategy if available
        if pattern.id in self.adaptation_strategies:
            return self.adaptation_strategies[pattern.id](pattern, assumptions, tools, count)
        
        # Otherwise use generic adaptation
        return self._generic_adaptation(pattern, assumptions, tools, count)
    
    def _adapt_precision_loss(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt precision loss pattern for financial calculations."""
        scenarios = []
        
        # Get currencies from assumptions or use defaults
        currencies = list(assumptions.currencies) if assumptions.currencies else ["USD", "EUR", "JPY"]
        
        # Generate currency conversion scenarios
        for i in range(count):
            # Create conversion chain
            curr_chain = random.sample(currencies, min(3, len(currencies)))
            if len(curr_chain) < 3:
                curr_chain.extend(random.choices(["GBP", "CHF", "CAD"], k=3-len(curr_chain)))
            
            amount = random.choice([0.01, 0.001, 9999999.99, 123456.789])
            
            task = f"Convert {amount} {curr_chain[0]} to {curr_chain[1]}, then to {curr_chain[2]}, and finally back to {curr_chain[0]}. Calculate the exact difference from the original amount."
            
            expected_tools = self._select_relevant_tools(tools, ["currency", "calculator", "converter"])
            
            expected_error = f"The currency conversion chain will lose precision due to floating-point rounding across multiple conversions. The final amount will differ from {amount} {curr_chain[0]} by approximately 0.01-0.001%, causing a mismatch in financial calculations that require exact precision."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "currencies": curr_chain,
                    "amount": amount
                }
            ))
        
        return scenarios
    
    def _adapt_malformed_response(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt malformed response pattern."""
        scenarios = []
        
        malformed_types = [
            ("HTML error page", "text/html", "JSON array"),
            ("XML response", "application/xml", "JSON object"),
            ("plain text", "text/plain", "structured data"),
            ("truncated JSON", "application/json", "complete response"),
            ("escaped JSON string", "application/json", "parsed object")
        ]
        
        for i in range(count):
            malformed = random.choice(malformed_types)
            search_term = random.choice(["", "null", "undefined", "<script>alert('test')</script>", "' OR 1=1 --"])
            
            task = f"Search for products with the query '{search_term}' and parse the results to extract product names and prices."
            
            expected_tools = self._select_relevant_tools(tools, ["search", "database", "api"])
            
            expected_error = f"The API will return {malformed[0]} with content-type {malformed[1]} instead of the expected {malformed[2]}, causing a parsing error. The response cannot be processed as structured data, resulting in a complete failure to extract product information."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "malformed_type": malformed[0],
                    "search_term": search_term
                }
            ))
        
        return scenarios
    
    def _adapt_timeout(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt timeout pattern."""
        scenarios = []
        
        timeout_scenarios = [
            ("999999 cities worldwide", "weather data", 30),
            ("all historical data since 1900", "stock prices", 60),
            ("infinite recursive search depth", "web pages", 120),
            ("real-time updates every millisecond", "sensor data", 10)
        ]
        
        for i in range(count):
            scenario = random.choice(timeout_scenarios)
            
            task = f"Retrieve {scenario[1]} for {scenario[0]} simultaneously and process all results within {scenario[2]} seconds."
            
            expected_tools = self._select_relevant_tools(tools, ["api", "http", "search", "database"])
            
            expected_error = f"The request will timeout after {scenario[2]} seconds due to attempting to fetch an unrealistic amount of data ({scenario[0]}). The API rate limiter or connection timeout will trigger, causing a 503 Service Unavailable or 504 Gateway Timeout error."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "data_scope": scenario[0],
                    "timeout_seconds": scenario[2]
                }
            ))
        
        return scenarios
    
    def _adapt_ambiguous_request(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt ambiguous request pattern."""
        scenarios = []
        
        ambiguous_params = [
            ("null location", "undefined time", "missing criteria"),
            ("", "now", "all"),
            ("here", "recently", "good ones"),
            ("that place", "sometime", "the usual")
        ]
        
        for i in range(count):
            params = random.choice(ambiguous_params)
            
            task = f"Check the weather at {params[0]} for {params[1]} and find products matching {params[2]}."
            
            expected_tools = self._select_relevant_tools(tools, ["weather", "search", "database"])
            
            expected_error = f"The request will fail due to ambiguous parameters: '{params[0]}' is not a valid location, '{params[1]}' is not a specific time, and '{params[2]}' provides no searchable criteria. APIs will return 400 Bad Request errors due to missing or invalid required parameters."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "ambiguous_params": params
                }
            ))
        
        return scenarios
    
    def _adapt_empty_results(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt empty results pattern."""
        scenarios = []
        
        impossible_queries = [
            "products priced at exactly negative infinity dollars",
            "flights departing yesterday arriving tomorrow",
            "restaurants open 25 hours a day",
            "users with null email addresses and empty names"
        ]
        
        for i in range(count):
            query = random.choice(impossible_queries)
            location = random.choice(["null island at 0°N 0°E", "undefined city", "coordinates -999,-999"])
            
            task = f"Find {query} near {location} and generate a detailed report."
            
            expected_tools = self._select_relevant_tools(tools, ["search", "database", "api"])
            
            expected_error = f"The search will return an empty result set because {query} is logically impossible and {location} is not a valid location. The API will return a 200 OK status but with an empty data array, causing downstream processing to fail when trying to generate a report from no data."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "impossible_query": query,
                    "invalid_location": location
                }
            ))
        
        return scenarios
    
    def _adapt_rate_limiting(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt rate limiting pattern."""
        scenarios = []
        
        # Get rate limits from assumptions or use defaults
        rate_limits = assumptions.rate_limits or {"default": 100}
        
        burst_scenarios = [
            ("1000 parallel requests", "user profiles"),
            ("continuous polling every 10ms", "status updates"),
            ("bulk operations on 10000 items", "database records"),
            ("real-time monitoring of 500 endpoints", "API health checks")
        ]
        
        for i in range(count):
            scenario = random.choice(burst_scenarios)
            limit = random.choice(list(rate_limits.values())) if rate_limits else 100
            
            task = f"Execute {scenario[0]} to fetch {scenario[1]} and process all responses immediately."
            
            expected_tools = self._select_relevant_tools(tools, ["api", "http", "database"])
            
            expected_error = f"The API will enforce rate limiting after approximately {limit} requests, returning 429 Too Many Requests errors. {scenario[0]} far exceeds the rate limit, causing the majority of requests to fail and preventing complete data collection."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "request_pattern": scenario[0],
                    "rate_limit": limit
                }
            ))
        
        return scenarios
    
    def _adapt_auth_error(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt authentication error pattern."""
        scenarios = []
        
        auth_scenarios = [
            ("expired token from 1970-01-01", "401 Unauthorized"),
            ("null authentication header", "401 Unauthorized"),
            ("invalid API key 'test123'", "403 Forbidden"),
            ("insufficient scope for admin operations", "403 Forbidden")
        ]
        
        for i in range(count):
            scenario = random.choice(auth_scenarios)
            operation = random.choice(["delete all records", "access private user data", "modify system settings"])
            
            task = f"Using {scenario[0]}, {operation} in the production database."
            
            expected_tools = self._select_relevant_tools(tools, ["api", "database", "http"])
            
            expected_error = f"The request will fail with {scenario[1]} because {scenario[0]} does not provide valid authentication. The API will reject the request before processing, preventing any {operation} from executing."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "auth_issue": scenario[0],
                    "error_code": scenario[1]
                }
            ))
        
        return scenarios
    
    def _adapt_resource_exhaustion(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Adapt resource exhaustion pattern."""
        scenarios = []
        
        resource_scenarios = [
            ("infinite loop generating Fibonacci numbers", "CPU"),
            ("loading 1TB file into memory", "RAM"),
            ("creating 1 million temporary files", "disk space"),
            ("opening 100000 network connections", "file descriptors")
        ]
        
        for i in range(count):
            scenario = random.choice(resource_scenarios)
            
            task = f"Execute {scenario[0]} and return the complete results."
            
            expected_tools = self._select_relevant_tools(tools, ["calculator", "file", "shell"])
            
            expected_error = f"The operation will exhaust available {scenario[1]} resources, causing the system to become unresponsive or crash. {scenario[0]} requires more resources than available, leading to an out-of-memory error or system timeout."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "resource_type": scenario[1],
                    "exhaustion_method": scenario[0]
                }
            ))
        
        return scenarios
    
    def _generic_adaptation(
        self,
        pattern: FailurePattern,
        assumptions: AgentAssumptions,
        tools: List[str],
        count: int
    ) -> List[ScenarioTemplate]:
        """Generic adaptation for patterns without specific strategies."""
        scenarios = []
        
        for i in range(count):
            # Build task with edge cases
            edge_cases = random.sample(self.edge_case_keywords, min(3, len(self.edge_case_keywords)))
            
            task_templates = [
                f"Process data with {edge_cases[0]} values and handle {edge_cases[1]} conditions",
                f"Search for items where the criteria is {edge_cases[0]} and the location is {edge_cases[1]}",
                f"Calculate results when input is {edge_cases[0]} and parameters are {edge_cases[1]}",
                f"Retrieve information with {edge_cases[0]} identifier from {edge_cases[1]} source"
            ]
            
            task = random.choice(task_templates)
            if len(edge_cases) > 2:
                task += f" while dealing with {edge_cases[2]} constraints"
            
            # Select tools based on category
            category_keywords = {
                "calculation": ["calculator", "compute", "math"],
                "data": ["database", "api", "search", "query"],
                "infrastructure": ["http", "api", "network", "service"],
                "logic": ["process", "workflow", "decision"]
            }
            keywords = category_keywords.get(pattern.category, ["api", "data"])
            expected_tools = self._select_relevant_tools(tools, keywords)
            
            # Build error description with proper indicators and technical details
            error_parts = []
            
            # Start with a specific error indicator
            indicators = ["will fail with", "causes", "triggers", "results in", "leads to"]
            error_parts.append(f"The operation {random.choice(indicators)} {pattern.title}")
            
            # Add specific technical details based on category
            if pattern.category == "infrastructure":
                error_parts.append("The API will return 500 Internal Server Error or 504 Gateway Timeout")
                error_parts.append(f"when attempting to process {edge_cases[0]} values with {edge_cases[1]} conditions")
            elif pattern.category == "data":
                error_parts.append("JSON parsing will fail with validation errors")
                error_parts.append(f"due to {edge_cases[0]} data combined with {edge_cases[1]} format constraints")
            elif pattern.category == "calculation":
                error_parts.append("Calculations will produce incorrect results with precision loss")
                error_parts.append(f"when handling {edge_cases[0]} numbers under {edge_cases[1]} conditions")
            elif pattern.category == "logic":
                error_parts.append("The system will encounter ambiguous logic flow")
                error_parts.append(f"when processing {edge_cases[0]} inputs with {edge_cases[1]} parameters")
            else:
                error_parts.append(f"Processing will fail when encountering {edge_cases[0]} values")
                error_parts.append(f"combined with {edge_cases[1]} conditions")
            
            # Add more technical context
            if len(edge_cases) > 2:
                error_parts.append(f"especially with {edge_cases[2]} constraints")
            
            # Add pattern-specific error if available
            if pattern.expected_error:
                error_parts.append(pattern.expected_error)
            
            expected_error = ". ".join(error_parts) + "."
            
            scenarios.append(ScenarioTemplate(
                task_prompt=task,
                expected_tools=expected_tools,
                expected_error=expected_error,
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_category": pattern.category,
                    "edge_cases": edge_cases
                }
            ))
        
        return scenarios
    
    def _select_relevant_tools(
        self,
        available_tools: List[str],
        preferred_keywords: List[str]
    ) -> List[str]:
        """Select relevant tools based on keywords."""
        if not available_tools:
            return []
        
        selected = []
        
        # First, try to find tools matching keywords
        for tool in available_tools:
            tool_lower = tool.lower()
            if any(keyword in tool_lower for keyword in preferred_keywords):
                selected.append(tool)
        
        # If no matches, select up to 2 random tools
        if not selected and available_tools:
            selected = random.sample(available_tools, min(2, len(available_tools)))
        
        # Limit to 3 tools max
        return selected[:3]