"""
Flexible agent configuration parser for Arc-CLI.

This parser is framework-agnostic and focuses on extracting capabilities
from any YAML structure rather than enforcing rigid formats.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class AgentConfigParser:
    """Parse agent configurations from various formats into normalized structure."""
    
    # Common field variations we'll recognize
    FIELD_MAPPINGS = {
        "model": ["model", "model_name", "llm", "llm_model", "ai_model"],
        "temperature": ["temperature", "temp", "creativity"],
        "system_prompt": ["system_prompt", "system_message", "instructions", "prompt", "role"],
        "tools": ["tools", "functions", "capabilities", "plugins", "actions"],
        "max_tokens": ["max_tokens", "max_length", "token_limit"],
        "top_p": ["top_p", "nucleus_sampling"],
        "assumptions": ["assumptions", "constraints", "rules", "guidelines"],
        "job": ["job", "job_description", "purpose", "objective", "mission"],
        "validation_rules": ["validation_rules", "validation", "rules", "checks"],
        "name": ["name", "agent_name", "assistant_name"],
        "description": ["description", "desc", "about"],
    }
    
    # Minimum required fields for a valid agent config
    REQUIRED_FIELDS = ["model", "temperature", "system_prompt", "tools"]
    
    def __init__(self):
        """Initialize the parser."""
        self.last_parsed_config = None
        self.warnings = []
    
    def parse(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse agent configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Normalized configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required fields
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        
        if not isinstance(raw_config, dict):
            raise ValueError("Configuration must be a YAML dictionary/object")
        
        # Normalize the configuration
        normalized = self._normalize_config(raw_config)
        
        # Validate required fields
        self._validate_config(normalized)
        
        # Store for reference
        self.last_parsed_config = normalized
        
        return normalized
    
    def _normalize_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration by mapping common field variations.
        
        Args:
            raw_config: Raw configuration dictionary
            
        Returns:
            Normalized configuration
        """
        normalized = {}
        self.warnings = []
        
        # Map fields using our mappings
        for standard_field, variations in self.FIELD_MAPPINGS.items():
            value = None
            for variation in variations:
                if variation in raw_config:
                    value = raw_config[variation]
                    break
            
            if value is not None:
                normalized[standard_field] = value
        
        # Handle tools specially - ensure it's a list
        if "tools" in normalized:
            normalized["tools"] = self._normalize_tools(normalized["tools"])
        
        # Preserve any additional fields that might be useful
        for key, value in raw_config.items():
            if key not in normalized and not self._is_known_variation(key):
                normalized[key] = value
                self.warnings.append(f"Unknown field '{key}' preserved as-is")
        
        # Apply defaults for optional fields
        normalized.setdefault("temperature", 0.7)
        normalized.setdefault("max_tokens", None)
        normalized.setdefault("top_p", 1.0)
        
        return normalized
    
    def _normalize_tools(self, tools: Any) -> List[Union[str, Dict[str, Any]]]:
        """
        Normalize tools preserving full definitions when available.
        
        Args:
            tools: Tools in various formats (list of strings, list of dicts, etc.)
            
        Returns:
            List of tool definitions (either strings or full dict definitions)
        """
        if not tools:
            return []
        
        if isinstance(tools, str):
            # Single tool as string
            return [tools]
        
        if not isinstance(tools, list):
            self.warnings.append(f"Unexpected tools format: {type(tools)}")
            return []
        
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, str):
                # Simple string tool name - preserve as is
                normalized_tools.append(tool)
            elif isinstance(tool, dict):
                # Tool as dictionary - preserve full definition
                if "name" in tool:
                    # Already has proper structure
                    normalized_tools.append(tool)
                elif "tool" in tool:
                    # Rename 'tool' to 'name' for consistency
                    tool_def = tool.copy()
                    tool_def["name"] = tool_def.pop("tool")
                    normalized_tools.append(tool_def)
                else:
                    # Try to use the first key as tool name
                    if tool:
                        tool_name = list(tool.keys())[0]
                        # Create a normalized structure
                        normalized_tools.append({
                            "name": tool_name,
                            "definition": tool[tool_name]
                        })
                        self.warnings.append(f"Restructured tool '{tool_name}' from nested format")
            else:
                self.warnings.append(f"Skipping tool with unexpected type: {type(tool)}")
        
        return normalized_tools
    
    def _is_known_variation(self, field: str) -> bool:
        """Check if a field is a known variation of a standard field."""
        for variations in self.FIELD_MAPPINGS.values():
            if field in variations:
                return True
        return False
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate that configuration has minimum required fields.
        
        Args:
            config: Normalized configuration
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if field not in config or config[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(missing_fields)}. "
                f"Required fields are: {', '.join(self.REQUIRED_FIELDS)}"
            )
        
        # Additional validation
        if not isinstance(config["tools"], list):
            raise ValueError("Tools must be a list")
        
        if not isinstance(config["temperature"], (int, float)):
            raise ValueError("Temperature must be a number")
        
        if config["temperature"] < 0 or config["temperature"] > 2:
            self.warnings.append(f"Temperature {config['temperature']} is outside typical range [0, 2]")
    
    def get_warnings(self) -> List[str]:
        """Get any warnings from the last parse operation."""
        return self.warnings.copy()
    
    def get_tool_names(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract just the tool names from the configuration.
        
        Args:
            config: Configuration to analyze (uses last parsed if not provided)
            
        Returns:
            List of tool names as strings
        """
        if config is None:
            config = self.last_parsed_config
            if config is None:
                raise ValueError("No configuration to analyze")
        
        tools = config.get("tools", [])
        tool_names = []
        
        for tool in tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            elif isinstance(tool, dict) and "name" in tool:
                tool_names.append(tool["name"])
        
        return tool_names
    
    def extract_capabilities(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract high-level capabilities from the configuration.
        
        Args:
            config: Configuration to analyze (uses last parsed if not provided)
            
        Returns:
            Dictionary of extracted capabilities
        """
        if config is None:
            config = self.last_parsed_config
            if config is None:
                raise ValueError("No configuration to analyze")
        
        # Get tool names for categorization
        tool_names = self.get_tool_names(config)
        
        capabilities = {
            "domains": self._infer_domains(config),
            "tool_categories": self._categorize_tools(tool_names),
            "behavioral_traits": self._extract_behavioral_traits(config),
            "complexity_level": self._estimate_complexity(config),
            "tools": config.get("tools", []),  # Include full tool definitions
            "assumptions": config.get("assumptions", []),
            "job": config.get("job", ""),
            "validation_rules": config.get("validation_rules", []),
        }
        
        return capabilities
    
    def _infer_domains(self, config: Dict[str, Any]) -> List[str]:
        """Infer likely domains from system prompt and tools."""
        domains = []
        
        prompt = config.get("system_prompt", "").lower()
        tool_names = [t.lower() for t in self.get_tool_names(config)]
        
        # Domain keywords mapping
        domain_indicators = {
            "finance": ["financial", "finance", "money", "currency", "trading", "investment", "accounting"],
            "healthcare": ["medical", "health", "patient", "diagnosis", "treatment", "clinical"],
            "education": ["teaching", "learning", "student", "curriculum", "educational", "training"],
            "customer_service": ["customer", "support", "help", "assist", "service", "inquiry"],
            "research": ["research", "analysis", "study", "investigate", "academic", "scientific"],
            "engineering": ["engineering", "technical", "develop", "build", "design", "architect"],
        }
        
        for domain, keywords in domain_indicators.items():
            if any(keyword in prompt for keyword in keywords):
                domains.append(domain)
            elif any(any(keyword in tool for keyword in keywords) for tool in tool_names):
                domains.append(domain)
        
        return list(set(domains)) or ["general"]
    
    def _categorize_tools(self, tools: List[str]) -> Dict[str, List[str]]:
        """Categorize tools by their likely function."""
        categories = {
            "search": [],
            "calculation": [],
            "data_processing": [],
            "external_api": [],
            "file_operations": [],
            "communication": [],
            "other": []
        }
        
        tool_patterns = {
            "search": ["search", "find", "query", "lookup"],
            "calculation": ["calc", "compute", "math", "sum", "average"],
            "data_processing": ["parse", "extract", "transform", "analyze", "aggregate"],
            "external_api": ["api", "fetch", "request", "webhook", "integration"],
            "file_operations": ["read", "write", "file", "save", "load"],
            "communication": ["email", "message", "notify", "send", "chat"],
        }
        
        for tool in tools:
            tool_lower = tool.lower()
            categorized = False
            
            for category, patterns in tool_patterns.items():
                if any(pattern in tool_lower for pattern in patterns):
                    categories[category].append(tool)
                    categorized = True
                    break
            
            if not categorized:
                categories["other"].append(tool)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _extract_behavioral_traits(self, config: Dict[str, Any]) -> List[str]:
        """Extract behavioral traits from configuration."""
        traits = []
        
        # Temperature-based traits
        temp = config.get("temperature", 0.7)
        if temp < 0.3:
            traits.append("highly_deterministic")
        elif temp > 1.2:
            traits.append("highly_creative")
        
        # System prompt analysis
        prompt = config.get("system_prompt", "").lower()
        if "concise" in prompt or "brief" in prompt:
            traits.append("concise_responses")
        if "detailed" in prompt or "comprehensive" in prompt:
            traits.append("detailed_responses")
        if "formal" in prompt or "professional" in prompt:
            traits.append("formal_tone")
        if "friendly" in prompt or "casual" in prompt:
            traits.append("casual_tone")
        
        # Assumption-based traits
        assumptions = config.get("assumptions", [])
        if assumptions:
            traits.append("assumption_driven")
        
        return traits
    
    def _estimate_complexity(self, config: Dict[str, Any]) -> str:
        """Estimate agent complexity based on configuration."""
        score = 0
        
        # Tool count
        tool_count = len(config.get("tools", []))
        if tool_count == 0:
            score += 0
        elif tool_count <= 3:
            score += 1
        elif tool_count <= 7:
            score += 2
        else:
            score += 3
        
        # System prompt length
        prompt_length = len(config.get("system_prompt", ""))
        if prompt_length < 100:
            score += 0
        elif prompt_length < 500:
            score += 1
        else:
            score += 2
        
        # Presence of advanced features
        if config.get("assumptions"):
            score += 1
        if config.get("max_tokens"):
            score += 1
        
        # Map score to complexity level
        if score <= 2:
            return "simple"
        elif score <= 5:
            return "moderate"
        else:
            return "complex"