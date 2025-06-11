"""
Tool Behavior Engine for Arc Sandbox

Dynamically creates realistic tool implementations based on agent tool definitions.
This engine can generate behaviors for ANY tool type, not just predefined ones.
"""

import json
import time
import random
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field, create_model
import re


class ToolBehaviorEngine:
    """
    Generates realistic tool behaviors dynamically based on agent definitions.
    
    Key Features:
    - Works with any tool definition without hardcoding
    - Generates appropriate responses based on tool description
    - Can inject failures based on assumptions and scenarios
    - Maintains realistic timing and error patterns
    """
    
    def __init__(self, agent_profile: Dict[str, Any], scenario_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool behavior engine.
        
        Args:
            agent_profile: Complete agent profile including tools, assumptions, etc.
            scenario_context: Current scenario being tested (for failure injection)
        """
        self.agent_profile = agent_profile
        self.scenario_context = scenario_context or {}
        self.agent_config = agent_profile.get("configuration", {})
        self.assumptions = agent_profile.get("assumptions", [])
        self.validation_rules = agent_profile.get("validation_rules", [])
        
        # Track tool usage for realistic behavior
        self.tool_call_history = []
        self.call_count = {}
        
        # Load or generate behavior patterns
        self.behavior_patterns = self._load_behavior_patterns()
    
    def create_tools(self) -> List[Any]:
        """
        Create LangChain tools from agent tool definitions.
        
        Returns:
            List of LangChain tool functions with realistic behaviors
        """
        tools = []
        tool_definitions = self.agent_config.get("tools", [])
        
        for tool_def in tool_definitions:
            if isinstance(tool_def, str):
                # Simple string tool - create basic implementation
                tool_func = self._create_generic_tool(tool_def, {})
            elif isinstance(tool_def, dict):
                # Full tool definition
                tool_func = self._create_tool_from_definition(tool_def)
            else:
                continue
            
            if tool_func:
                tools.append(tool_func)
        
        return tools
    
    def _create_tool_from_definition(self, tool_def: Dict[str, Any]) -> Optional[Any]:
        """Create a LangChain tool from a tool definition."""
        tool_name = tool_def.get("name", "unknown_tool")
        description = tool_def.get("description", f"Tool: {tool_name}")
        
        # Create Pydantic model for tool inputs dynamically
        input_schema = self._create_input_schema(tool_def)
        
        # Create the tool implementation
        def tool_implementation(**kwargs) -> str:
            """Dynamic tool implementation based on definition."""
            return self._execute_tool_behavior(tool_name, tool_def, kwargs)
        
        # Create LangChain tool with proper schema
        if input_schema:
            @tool(args_schema=input_schema)
            def wrapped_tool(**kwargs) -> str:
                """Dynamic tool implementation."""
                return tool_implementation(**kwargs)
            
            # Update tool metadata
            wrapped_tool.name = tool_name
            wrapped_tool.description = description
            
            return wrapped_tool
        else:
            # No schema - simple tool
            @tool
            def wrapped_tool(input: str = "") -> str:
                """Dynamic tool implementation."""
                return tool_implementation(input=input)
            
            wrapped_tool.name = tool_name
            wrapped_tool.description = description
            
            return wrapped_tool
    
    def _create_input_schema(self, tool_def: Dict[str, Any]) -> Optional[type]:
        """Create a Pydantic model for tool inputs dynamically."""
        # Extract parameters from tool definition
        required_params = tool_def.get("required_params", [])
        optional_params = tool_def.get("optional_params", [])
        
        if not required_params and not optional_params:
            # Check for other parameter formats
            if "parameters" in tool_def:
                return self._create_schema_from_parameters(tool_def["parameters"])
            return None
        
        # Build field definitions
        fields = {}
        
        # Process required parameters
        for param in required_params:
            if isinstance(param, dict):
                # Format: {"param_name": "description"}
                for name, desc in param.items():
                    fields[name] = (str, Field(description=desc))
            elif isinstance(param, str):
                # Simple string parameter name
                fields[param] = (str, Field(description=f"Parameter: {param}"))
        
        # Process optional parameters
        for param in optional_params:
            if isinstance(param, dict):
                for name, desc in param.items():
                    fields[name] = (Optional[str], Field(default=None, description=desc))
            elif isinstance(param, str):
                fields[param] = (Optional[str], Field(default=None, description=f"Optional: {param}"))
        
        if not fields:
            return None
        
        # Create dynamic Pydantic model
        model_name = f"{tool_def.get('name', 'Tool')}Input"
        return create_model(model_name, **fields)
    
    def _create_schema_from_parameters(self, parameters: Dict[str, Any]) -> Optional[type]:
        """Create schema from OpenAPI-style parameters."""
        if not isinstance(parameters, dict):
            return None
        
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        fields = {}
        for name, prop in properties.items():
            prop_type = prop.get("type", "string")
            desc = prop.get("description", f"Parameter: {name}")
            
            # Map JSON schema types to Python types
            type_mapping = {
                "string": str,
                "number": float,
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            py_type = type_mapping.get(prop_type, str)
            
            if name in required:
                fields[name] = (py_type, Field(description=desc))
            else:
                fields[name] = (Optional[py_type], Field(default=None, description=desc))
        
        if not fields:
            return None
        
        return create_model("ToolInput", **fields)
    
    def _execute_tool_behavior(self, tool_name: str, tool_def: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """
        Execute realistic tool behavior based on tool definition and context.
        
        This is where the magic happens - we generate appropriate responses
        based on the tool's purpose without hardcoding specific behaviors.
        """
        # Track call
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        call_info = {
            "tool": tool_name,
            "inputs": inputs,
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count[tool_name]
        }
        self.tool_call_history.append(call_info)
        
        # Simulate realistic response time
        response_time = random.uniform(0.1, 0.5)
        time.sleep(response_time)
        
        # Check if we should inject a failure
        if self._should_fail(tool_name, inputs):
            return self._generate_failure(tool_name, tool_def, inputs)
        
        # Generate successful response based on tool type
        return self._generate_success_response(tool_name, tool_def, inputs)
    
    def _should_fail(self, tool_name: str, inputs: Dict[str, Any]) -> bool:
        """Determine if this tool call should fail based on scenario context."""
        # Check if scenario expects this tool to fail
        scenario = self.scenario_context
        expected_failures = scenario.get("expected_failures", [])
        
        if tool_name in expected_failures:
            return True
        
        # Check for assumption violations
        if self._violates_assumptions(tool_name, inputs):
            return True
        
        # Random failure rate (simulate real-world unreliability)
        base_failure_rate = 0.02  # 2% baseline
        if scenario.get("complexity_level") == "complex":
            base_failure_rate = 0.05  # Higher failure rate for complex scenarios
        
        return random.random() < base_failure_rate
    
    def _violates_assumptions(self, tool_name: str, inputs: Dict[str, Any]) -> bool:
        """Check if inputs violate agent assumptions."""
        # This is where we test assumption violations dynamically
        
        # Example: Check currency assumptions for finance tools
        if any("currency" in assumption.lower() for assumption in self.assumptions):
            # If agent assumes USD but input suggests otherwise
            input_str = json.dumps(inputs).lower()
            if any(currency in input_str for currency in ["eur", "gbp", "jpy", "€", "£", "¥"]):
                # Input contains non-USD currency but agent assumes USD
                if "usd" in " ".join(self.assumptions).lower():
                    return True
        
        # Check for format assumptions (dates, numbers, etc.)
        if "date" in tool_name.lower() or any("date" in str(v).lower() for v in inputs.values()):
            # Check date format assumptions
            date_patterns = {
                "mm/dd/yyyy": r"\d{2}/\d{2}/\d{4}",
                "dd/mm/yyyy": r"\d{2}/\d{2}/\d{4}",
                "yyyy-mm-dd": r"\d{4}-\d{2}-\d{2}"
            }
            # Could check if date format matches assumptions
        
        return False
    
    def _generate_failure(self, tool_name: str, tool_def: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Generate a realistic failure response."""
        failure_type = self._determine_failure_type(tool_name, tool_def)
        
        failure_responses = {
            "validation_error": {
                "error": "Validation failed",
                "message": f"Invalid input for {tool_name}: {self._get_validation_error(inputs)}",
                "code": "VALIDATION_ERROR"
            },
            "not_found": {
                "error": "Resource not found",
                "message": f"The requested resource could not be found",
                "code": "NOT_FOUND"
            },
            "timeout": {
                "error": "Request timeout",
                "message": f"Operation timed out after 30 seconds",
                "code": "TIMEOUT"
            },
            "rate_limit": {
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60,
                "code": "RATE_LIMITED"
            },
            "assumption_violation": {
                "error": "Assumption violation",
                "message": self._get_assumption_violation_message(tool_name, inputs),
                "code": "ASSUMPTION_VIOLATED"
            }
        }
        
        response = failure_responses.get(failure_type, failure_responses["validation_error"])
        return json.dumps(response, indent=2)
    
    def _generate_success_response(self, tool_name: str, tool_def: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Generate a realistic success response based on tool purpose."""
        description = tool_def.get("description", "").lower()
        
        # Analyze tool purpose from description
        if any(word in description for word in ["spreadsheet", "excel", "csv", "data", "analyze"]):
            return self._generate_data_analysis_response(tool_name, inputs)
        elif any(word in description for word in ["calculate", "compute", "ratio", "formula"]):
            return self._generate_calculation_response(tool_name, inputs)
        elif any(word in description for word in ["query", "search", "database", "retrieve"]):
            return self._generate_query_response(tool_name, inputs)
        elif any(word in description for word in ["report", "generate", "format", "pdf"]):
            return self._generate_report_response(tool_name, inputs)
        elif any(word in description for word in ["weather", "temperature", "forecast"]):
            return self._generate_weather_response(tool_name, inputs)
        else:
            # Generic response for unknown tool types
            return self._generate_generic_response(tool_name, inputs)
    
    def _generate_data_analysis_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate response for data analysis tools."""
        response = {
            "status": "success",
            "data": {
                "rows_processed": random.randint(100, 10000),
                "columns": ["Date", "Amount", "Category", "Description"],
                "summary": {
                    "total": round(random.uniform(10000, 1000000), 2),
                    "average": round(random.uniform(100, 10000), 2),
                    "count": random.randint(50, 500)
                },
                "patterns_found": [
                    "Monthly recurring transactions detected",
                    "Seasonal variation in Q4"
                ]
            },
            "metadata": {
                "file": inputs.get("file_path", "data.csv"),
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Add currency data if this is a finance-related analysis
        if any(word in tool_name.lower() for word in ["finance", "financial", "money", "currency"]):
            # Check if agent assumes USD
            assumes_usd = any("usd" in assumption.lower() for assumption in self.assumptions)
            if assumes_usd:
                response["data"]["currency"] = "USD"  # Always return USD if assumed
            else:
                response["data"]["currency"] = "MIXED"  # Might have multiple currencies
        
        return json.dumps(response, indent=2)
    
    def _generate_calculation_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate response for calculation tools."""
        calc_type = inputs.get("calculation_type", "basic")
        values = inputs.get("values", [])
        
        response = {
            "status": "success",
            "calculation_type": calc_type,
            "result": round(random.uniform(0.5, 2.5) * sum([float(v) for v in str(values).split() if v.replace('.','').isdigit()]), 2) if values else random.uniform(100, 1000),
            "formula_used": f"{calc_type.upper()} calculation",
            "confidence": 0.95,
            "units": "undefined"  # This could cause issues if not handled
        }
        
        # Add assumptions made during calculation
        if "currency" in str(inputs).lower() or "financial" in tool_name.lower():
            response["assumptions_made"] = ["All values assumed to be in USD"]
        
        return json.dumps(response, indent=2)
    
    def _generate_query_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate response for query/database tools."""
        query_type = inputs.get("query_type", "general")
        
        response = {
            "status": "success",
            "query_type": query_type,
            "results": [
                {
                    "id": f"REC_{random.randint(1000, 9999)}",
                    "timestamp": datetime.now().isoformat(),
                    "value": round(random.uniform(100, 10000), 2),
                    "metadata": {"source": "primary_db"}
                }
                for _ in range(random.randint(1, 5))
            ],
            "total_records": random.randint(1, 100),
            "query_time_ms": random.randint(50, 500)
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_report_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate response for report generation tools."""
        report_type = inputs.get("report_type", "summary")
        
        response = {
            "status": "success",
            "report_type": report_type,
            "file_generated": f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "sections": [
                "Executive Summary",
                "Detailed Analysis", 
                "Recommendations",
                "Appendix"
            ],
            "page_count": random.randint(5, 25),
            "generation_time_seconds": round(random.uniform(2, 10), 1)
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_weather_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate response for weather tools."""
        city = inputs.get("city", "Unknown")
        
        response = {
            "location": city,
            "current": {
                "temperature": random.randint(50, 90),
                "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]),
                "humidity": random.randint(30, 80),
                "wind_speed": random.randint(5, 25)
            },
            "forecast": "Conditions expected to remain stable",
            "units": "fahrenheit"  # Could be celsius in other regions
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_generic_response(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate generic response for unknown tool types."""
        response = {
            "status": "success",
            "tool": tool_name,
            "message": f"Operation completed successfully",
            "data": {
                "input_received": list(inputs.keys()),
                "processing_time_ms": random.randint(100, 1000),
                "result_id": f"RESULT_{random.randint(10000, 99999)}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(response, indent=2)
    
    def _determine_failure_type(self, tool_name: str, tool_def: Dict[str, Any]) -> str:
        """Determine appropriate failure type based on context."""
        # Check scenario context for specific failure mode
        if self.scenario_context.get("failure_mode"):
            return self.scenario_context["failure_mode"]
        
        # Check if this is an assumption violation
        if self._violates_assumptions(tool_name, {}):
            return "assumption_violation"
        
        # Random failure type
        failure_types = ["validation_error", "not_found", "timeout", "rate_limit"]
        return random.choice(failure_types)
    
    def _get_validation_error(self, inputs: Dict[str, Any]) -> str:
        """Generate appropriate validation error message."""
        if not inputs:
            return "Required parameters missing"
        
        # Check for common validation issues
        for key, value in inputs.items():
            if value is None or value == "":
                return f"Parameter '{key}' cannot be empty"
            if "date" in key.lower() and not self._is_valid_date_format(str(value)):
                return f"Invalid date format for '{key}'"
            if "currency" in key.lower() and len(str(value)) != 3:
                return f"Invalid currency code for '{key}'"
        
        return "Input validation failed"
    
    def _get_assumption_violation_message(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate message for assumption violations."""
        # Check which assumption was violated
        for assumption in self.assumptions:
            if "currency" in assumption.lower() and "usd" in assumption.lower():
                # Check if non-USD currency in inputs
                input_str = json.dumps(inputs).lower()
                if any(curr in input_str for curr in ["eur", "gbp", "jpy"]):
                    return f"Tool assumes {assumption} but received non-USD currency data"
            
            if "date" in assumption.lower():
                # Date format mismatch
                if "mm/dd/yyyy" in assumption.lower():
                    return f"Tool assumes {assumption} but received different date format"
        
        return "Input violates tool assumptions"
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if date string matches expected formats."""
        patterns = [
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}-\d{2}-\d{4}"   # DD-MM-YYYY
        ]
        return any(re.match(pattern, date_str) for pattern in patterns)
    
    def _load_behavior_patterns(self) -> Dict[str, Any]:
        """Load or generate behavior patterns for tools."""
        # In a full implementation, this could load from a JSON file
        # For now, return basic patterns
        return {
            "response_time_ranges": {
                "fast": (0.05, 0.2),
                "normal": (0.2, 1.0),
                "slow": (1.0, 5.0)
            },
            "failure_rates": {
                "stable": 0.01,
                "normal": 0.02,
                "unstable": 0.1
            }
        }
    
    def _create_generic_tool(self, tool_name: str, metadata: Dict[str, Any]) -> Any:
        """Create a generic tool with basic behavior."""
        @tool
        def generic_tool(input: str = "") -> str:
            """Generic tool implementation."""
            return self._execute_tool_behavior(
                tool_name,
                {"name": tool_name, "description": f"Generic tool: {tool_name}"},
                {"input": input}
            )
        
        generic_tool.name = tool_name
        generic_tool.description = f"Tool: {tool_name}"
        
        return generic_tool