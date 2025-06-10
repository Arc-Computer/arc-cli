"""
Configuration normalizer for Arc-CLI.

Handles the transformation of parsed configurations into Arc's internal format
and provides utilities for working with normalized configurations.
"""

from typing import Dict, Any, List, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class ConfigNormalizer:
    """Normalize and enhance agent configurations for Arc processing."""
    
    def __init__(self):
        """Initialize the normalizer."""
        self.enhancements_applied = []
    
    def normalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration to Arc's internal format.
        
        Args:
            config: Parsed configuration dictionary
            
        Returns:
            Normalized configuration ready for Arc processing
        """
        # Deep copy to avoid modifying original
        normalized = copy.deepcopy(config)
        self.enhancements_applied = []
        
        # Apply normalization steps
        normalized = self._normalize_model_names(normalized)
        normalized = self._normalize_tool_definitions(normalized)
        normalized = self._add_arc_metadata(normalized)
        normalized = self._ensure_consistent_types(normalized)
        
        return normalized
    
    def _normalize_model_names(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize model names to consistent format.
        
        Maps various model naming conventions to standardized names.
        """
        model = config.get("model", "")
        
        # Model name mappings (lowercase -> standardized)
        # Based on experiments/generation/generator.py MODELS_TO_TEST
        model_mappings = {
            # OpenAI
            "gpt-4": "openai/gpt-4.1",
            "gpt4.1": "openai/gpt-4.1",
            "gpt-4.1": "openai/gpt-4.1",
            "gpt-4-turbo": "openai/gpt-4.1",
            "gpt-4o": "openai/gpt-4.1",
            "gpt-4o-mini": "openai/gpt-4.1-mini",
            "gpt-4.1-mini": "openai/gpt-4.1-mini",
            "gpt-3.5-turbo": "gpt-3.5-turbo",  # Keep as-is for backward compatibility
            
            # Anthropic
            "claude-3-opus": "anthropic/claude-opus-4",
            "claude-opus-4": "anthropic/claude-opus-4",
            "claude-3-sonnet": "anthropic/claude-sonnet-4",
            "claude-sonnet-4": "anthropic/claude-sonnet-4",
            "claude-3.5-sonnet": "anthropic/claude-sonnet-4",
            "claude-3-haiku": "anthropic/claude-3.5-haiku",
            "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
            
            # Google
            "gemini-pro": "google/gemini-2.5-pro-preview",
            "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
            "gemini-flash": "google/gemini-2.5-flash-preview-05-20",
            "gemini-2.5-flash": "google/gemini-2.5-flash-preview-05-20",
            "bard": "google/gemini-2.5-pro-preview",
            
            # Meta (Llama 4)
            "llama-3": "meta-llama/llama-4-scout",
            "llama3": "meta-llama/llama-4-scout",
            "llama-4-scout": "meta-llama/llama-4-scout",
            "llama-3-70b": "meta-llama/llama-4-maverick",
            "llama-4-maverick": "meta-llama/llama-4-maverick",
            
            # Mistral
            "mistral-medium": "mistralai/mistral-medium-3",
            "mistral-medium-3": "mistralai/mistral-medium-3",
            
            # Cohere
            "command": "cohere/command-a",
            "command-a": "cohere/command-a",
            
            # DeepSeek
            "deepseek-chat": "deepseek/deepseek-chat-v3-0324",
            "deepseek-chat-v3": "deepseek/deepseek-chat-v3-0324",
            "deepseek-r1": "deepseek/deepseek-r1-0528",
        }
        
        normalized_model = model_mappings.get(model.lower(), model)
        if normalized_model != model:
            config["model"] = normalized_model
            self.enhancements_applied.append(f"Normalized model name: {model} -> {normalized_model}")
        
        return config
    
    def _normalize_tool_definitions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool definitions to consistent format.
        
        Ensures tools are properly structured for Arc processing.
        """
        tools = config.get("tools", [])
        if not tools:
            return config
        
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, str):
                # Simple string - convert to structured format
                normalized_tools.append({
                    "name": tool,
                    "type": "function",
                    "enabled": True
                })
            elif isinstance(tool, dict):
                # Already structured - ensure required fields
                normalized_tool = {
                    "name": tool.get("name", tool.get("tool", "unknown")),
                    "type": tool.get("type", "function"),
                    "enabled": tool.get("enabled", True)
                }
                # Preserve any additional fields
                for key, value in tool.items():
                    if key not in ["name", "type", "enabled"]:
                        normalized_tool[key] = value
                normalized_tools.append(normalized_tool)
            else:
                # Skip invalid tools
                logger.warning(f"Skipping invalid tool: {tool}")
        
        config["tools_normalized"] = normalized_tools
        
        # Keep original tools list for compatibility
        if config["tools"] and not isinstance(config["tools"][0], str):
            # Extract just the names for the simple list
            config["tools"] = [t["name"] for t in normalized_tools]
            self.enhancements_applied.append("Normalized tool definitions to structured format")
        
        return config
    
    def _add_arc_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add Arc-specific metadata to configuration."""
        if "arc_metadata" not in config:
            config["arc_metadata"] = {
                "version": "1.0",
                "profile_type": "agent",
                "normalization_applied": True,
                "capability_analysis_pending": True
            }
            self.enhancements_applied.append("Added Arc metadata")
        
        return config
    
    def _ensure_consistent_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all fields have consistent types."""
        # Ensure numeric fields are actually numbers
        if "temperature" in config:
            try:
                config["temperature"] = float(config["temperature"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid temperature value: {config['temperature']}, using default 0.7")
                config["temperature"] = 0.7
        
        if "max_tokens" in config and config["max_tokens"] is not None:
            try:
                config["max_tokens"] = int(config["max_tokens"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid max_tokens value: {config['max_tokens']}, setting to None")
                config["max_tokens"] = None
        
        if "top_p" in config:
            try:
                config["top_p"] = float(config["top_p"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid top_p value: {config['top_p']}, using default 1.0")
                config["top_p"] = 1.0
        
        # Ensure lists are lists
        for field in ["tools", "assumptions"]:
            if field in config and not isinstance(config[field], list):
                if config[field] is None:
                    config[field] = []
                else:
                    config[field] = [config[field]]
                self.enhancements_applied.append(f"Converted {field} to list")
        
        return config
    
    def create_arc_profile(self, config: Dict[str, Any], capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete Arc agent profile from normalized config and capabilities.
        
        Args:
            config: Normalized configuration
            capabilities: Extracted capabilities from parser
            
        Returns:
            Complete Arc agent profile
        """
        profile = {
            "configuration": config,
            "capabilities": capabilities,
            "test_parameters": self._generate_test_parameters(config, capabilities),
            "optimization_targets": self._identify_optimization_targets(config, capabilities),
        }
        
        return profile
    
    def _generate_test_parameters(self, config: Dict[str, Any], capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommended test parameters based on agent profile."""
        # Base parameters
        params = {
            "scenario_count": 50,
            "parallel_execution": True,
            "max_workers": 10,
            "timeout_seconds": 30,
        }
        
        # Adjust based on complexity
        complexity = capabilities.get("complexity_level", "moderate")
        if complexity == "simple":
            params["scenario_count"] = 30
            params["timeout_seconds"] = 20
        elif complexity == "complex":
            params["scenario_count"] = 100
            params["timeout_seconds"] = 60
            params["max_workers"] = 20
        
        # Adjust based on domains
        domains = capabilities.get("domains", [])
        if "finance" in domains or "healthcare" in domains:
            params["require_deterministic"] = True
            params["statistical_validation"] = True
        
        # Tool-specific adjustments
        tool_categories = capabilities.get("tool_categories", {})
        if "external_api" in tool_categories:
            params["mock_external_calls"] = True
            params["rate_limit_aware"] = True
        
        return params
    
    def _identify_optimization_targets(self, config: Dict[str, Any], capabilities: Dict[str, Any]) -> List[str]:
        """Identify potential optimization targets for the agent."""
        targets = []
        
        # Temperature optimization
        if config.get("temperature", 0.7) > 1.0:
            targets.append("temperature_reduction")
        
        # Model optimization
        model = config.get("model", "")
        # Check for expensive models (with or without provider prefix)
        if "gpt-4.1" in model or "claude-opus-4" in model:
            targets.append("cost_optimization")
        
        # Tool optimization
        tools = config.get("tools", [])
        if len(tools) > 10:
            targets.append("tool_consolidation")
        
        # Domain-specific optimizations
        domains = capabilities.get("domains", [])
        if "finance" in domains:
            targets.append("numerical_precision")
        if "customer_service" in domains:
            targets.append("response_time")
        
        return targets or ["general_reliability"]
    
    def get_enhancements_applied(self) -> List[str]:
        """Get list of enhancements applied during normalization."""
        return self.enhancements_applied.copy()
    
    def validate_normalized_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate that a configuration is properly normalized.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["model", "temperature", "system_prompt", "tools", "arc_metadata"]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in normalized config: {field}")
                return False
        
        # Type checks
        if not isinstance(config["temperature"], (int, float)):
            logger.error("Temperature must be numeric")
            return False
        
        if not isinstance(config["tools"], list):
            logger.error("Tools must be a list")
            return False
        
        if not isinstance(config["arc_metadata"], dict):
            logger.error("Arc metadata must be a dictionary")
            return False
        
        return True