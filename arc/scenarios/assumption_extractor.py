"""
Extract assumptions from agent configurations for targeted scenario generation.
"""

import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class AgentAssumptions:
    """Extracted assumptions from agent configuration."""
    
    # Core assumptions
    tools: List[str] = field(default_factory=list)
    capabilities: Set[str] = field(default_factory=set)
    constraints: Set[str] = field(default_factory=set)
    
    # Domain-specific assumptions
    currencies: Set[str] = field(default_factory=set)
    data_formats: Set[str] = field(default_factory=set)
    api_versions: Set[str] = field(default_factory=set)
    
    # Behavioral assumptions
    error_handling: Set[str] = field(default_factory=set)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    timeouts: Dict[str, float] = field(default_factory=dict)
    
    # Environmental assumptions
    regions: Set[str] = field(default_factory=set)
    languages: Set[str] = field(default_factory=set)
    timezones: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tools": self.tools,
            "capabilities": list(self.capabilities),
            "constraints": list(self.constraints),
            "currencies": list(self.currencies),
            "data_formats": list(self.data_formats),
            "api_versions": list(self.api_versions),
            "error_handling": list(self.error_handling),
            "rate_limits": self.rate_limits,
            "timeouts": self.timeouts,
            "regions": list(self.regions),
            "languages": list(self.languages),
            "timezones": list(self.timezones)
        }


class AssumptionExtractor:
    """Extract assumptions from agent configurations."""
    
    def __init__(self):
        """Initialize the extractor with pattern matchers."""
        # Currency patterns
        self.currency_patterns = [
            r'\b(USD|EUR|GBP|JPY|CNY|CAD|AUD|CHF)\b',
            r'currency["\']?\s*[:=]\s*["\']?([A-Z]{3})',
            r'default_currency["\']?\s*[:=]\s*["\']?([A-Z]{3})',
        ]
        
        # API version patterns
        self.api_version_patterns = [
            r'api_version["\']?\s*[:=]\s*["\']?([v\d.]+)',
            r'version["\']?\s*[:=]\s*["\']?([v\d.]+)',
            r'/v(\d+)/',
        ]
        
        # Timeout patterns
        self.timeout_patterns = [
            r'timeout["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'max_wait["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'deadline["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
        ]
        
        # Rate limit patterns
        self.rate_limit_patterns = [
            r'rate_limit["\']?\s*[:=]\s*(\d+)',
            r'max_requests["\']?\s*[:=]\s*(\d+)',
            r'requests_per_[a-z]+["\']?\s*[:=]\s*(\d+)',
        ]
        
        # Data format patterns
        self.data_format_patterns = [
            r'format["\']?\s*[:=]\s*["\']?(json|xml|csv|yaml|html)',
            r'content[_-]type["\']?\s*[:=]\s*["\']?application/(\w+)',
            r'response[_-]format["\']?\s*[:=]\s*["\']?(\w+)',
        ]
    
    def extract(self, config: Dict[str, Any]) -> AgentAssumptions:
        """Extract assumptions from agent configuration.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            Extracted assumptions
        """
        assumptions = AgentAssumptions()
        
        # Extract tools
        tools = config.get('tools', [])
        if tools:
            # Handle different tool formats
            if isinstance(tools[0], dict):
                # Tools are dictionaries with 'name' field
                assumptions.tools = [tool.get('name', str(tool)) for tool in tools]
            else:
                # Tools are strings
                assumptions.tools = tools
        
        # Extract from system prompt
        system_prompt = config.get('system_prompt', '')
        if system_prompt:
            self._extract_from_text(system_prompt, assumptions)
        
        # Extract from tool descriptions
        tool_descriptions = config.get('tool_descriptions', {})
        if tool_descriptions:
            for tool_name, description in tool_descriptions.items():
                if isinstance(description, str):
                    self._extract_from_text(description, assumptions)
                elif isinstance(description, dict):
                    # Handle structured tool descriptions
                    self._extract_from_dict(description, assumptions)
        
        # Also extract from tools if they have descriptions
        if tools and isinstance(tools[0], dict):
            for tool in tools:
                if 'description' in tool:
                    self._extract_from_text(tool['description'], assumptions)
        
        # Extract from model config
        model_config = config.get('model_config', {})
        if model_config:
            self._extract_from_dict(model_config, assumptions)
        
        # Infer capabilities from tools
        self._infer_capabilities(assumptions)
        
        # Add default assumptions based on domain
        self._add_domain_defaults(config, assumptions)
        
        return assumptions
    
    def _extract_from_text(self, text: str, assumptions: AgentAssumptions) -> None:
        """Extract assumptions from text content."""
        text_lower = text.lower()
        
        # Extract currencies
        for pattern in self.currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assumptions.currencies.update(m.upper() if isinstance(m, str) else m[0].upper() for m in matches)
        
        # Extract API versions
        for pattern in self.api_version_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assumptions.api_versions.update(matches)
        
        # Extract timeouts
        for pattern in self.timeout_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                assumptions.timeouts[f"timeout_{len(assumptions.timeouts)}"] = float(match)
        
        # Extract rate limits
        for pattern in self.rate_limit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                assumptions.rate_limits[f"limit_{len(assumptions.rate_limits)}"] = int(match)
        
        # Extract data formats
        for pattern in self.data_format_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assumptions.data_formats.update(matches)
        
        # Extract error handling keywords
        error_keywords = ['retry', 'fallback', 'recover', 'handle error', 'exception']
        for keyword in error_keywords:
            if keyword in text_lower:
                assumptions.error_handling.add(keyword)
        
        # Extract regions
        region_patterns = ['us-east', 'us-west', 'eu-', 'asia-', 'global']
        for pattern in region_patterns:
            if pattern in text_lower:
                assumptions.regions.add(pattern)
        
        # Extract language references
        language_patterns = ['english', 'spanish', 'french', 'german', 'chinese', 'japanese']
        for lang in language_patterns:
            if lang in text_lower:
                assumptions.languages.add(lang)
    
    def _extract_from_dict(self, data: Dict[str, Any], assumptions: AgentAssumptions) -> None:
        """Extract assumptions from dictionary data."""
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check for specific keys
            if 'currency' in key_lower and isinstance(value, str):
                assumptions.currencies.add(value.upper())
            
            elif 'timeout' in key_lower and isinstance(value, (int, float)):
                assumptions.timeouts[key] = float(value)
            
            elif 'rate' in key_lower and 'limit' in key_lower and isinstance(value, (int, float)):
                assumptions.rate_limits[key] = int(value)
            
            elif 'format' in key_lower and isinstance(value, str):
                assumptions.data_formats.add(value.lower())
            
            elif 'version' in key_lower and isinstance(value, str):
                assumptions.api_versions.add(value)
            
            # Recursively extract from nested dicts
            elif isinstance(value, dict):
                self._extract_from_dict(value, assumptions)
            
            # Extract from lists of strings
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        self._extract_from_text(item, assumptions)
                    elif isinstance(item, dict):
                        self._extract_from_dict(item, assumptions)
    
    def _infer_capabilities(self, assumptions: AgentAssumptions) -> None:
        """Infer capabilities from tools."""
        tool_capabilities = {
            # Data tools
            "search": ["data_retrieval", "web_access"],
            "database": ["data_query", "structured_data"],
            "file": ["file_access", "data_storage"],
            
            # Calculation tools
            "calculator": ["arithmetic", "computation"],
            "currency": ["currency_conversion", "financial_calculation"],
            
            # Communication tools
            "email": ["communication", "notification"],
            "chat": ["conversation", "interaction"],
            
            # Navigation tools
            "browser": ["web_navigation", "dom_interaction"],
            "scraper": ["data_extraction", "web_parsing"],
            
            # API tools
            "api": ["external_integration", "data_exchange"],
            "http": ["web_requests", "api_calls"],
        }
        
        for tool in assumptions.tools:
            tool_lower = tool.lower()
            
            # Direct matches
            if tool_lower in tool_capabilities:
                assumptions.capabilities.update(tool_capabilities[tool_lower])
            
            # Partial matches
            for key, caps in tool_capabilities.items():
                if key in tool_lower or tool_lower in key:
                    assumptions.capabilities.update(caps)
    
    def _add_domain_defaults(self, config: Dict[str, Any], assumptions: AgentAssumptions) -> None:
        """Add default assumptions based on domain."""
        # Check for finance domain
        finance_indicators = ['finance', 'trading', 'investment', 'currency', 'stock', 'portfolio']
        system_prompt = config.get('system_prompt', '').lower()
        
        is_finance = any(indicator in system_prompt for indicator in finance_indicators)
        # Check if any tool contains finance indicators
        for tool in assumptions.tools:
            # Handle both string tools and dict tools
            tool_name = tool if isinstance(tool, str) else str(tool)
            if any(indicator in tool_name.lower() for indicator in finance_indicators):
                is_finance = True
                break
        
        if is_finance:
            # Add common finance assumptions
            if not assumptions.currencies:
                assumptions.currencies.add("USD")  # Default to USD if no currency specified
            
            assumptions.capabilities.add("financial_analysis")
            assumptions.data_formats.add("json")
            assumptions.error_handling.add("retry")  # Financial APIs often need retries
            
            # Add common financial constraints
            assumptions.constraints.add("regulatory_compliance")
            assumptions.constraints.add("precision_required")
        
        # Add general defaults
        if not assumptions.data_formats:
            assumptions.data_formats.add("json")  # Most common format
        
        if not assumptions.error_handling:
            assumptions.error_handling.add("basic")  # Minimal error handling
        
        if not assumptions.regions:
            assumptions.regions.add("us-east-1")  # Common default region